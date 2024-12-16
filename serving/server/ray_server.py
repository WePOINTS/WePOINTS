import ray
import logging
import torch
from PIL import Image
import base64
from io import BytesIO
from typing import Any, Callable, Literal
import contextlib
from pydantic import BaseModel
import fastapi
import math
import re
import aiohttp
import sys

logger = logging.getLogger(__name__)


@ray.remote
class WePOINTSModel:

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from wepoints.utils.images import Qwen2ImageProcessorForPOINTSV15

        model_path = "WePOINTS/POINTS-1-5-Qwen-2-5-7B-Chat"
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map='cuda')
        self._tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                        trust_remote_code=True)
        self._image_processor = Qwen2ImageProcessorForPOINTSV15.from_pretrained(
            model_path)

        def _construct_prompt(
            _self, messages: list[dict], image_processor
        ) -> tuple[str, list[Image.Image], list[list]]:  # noqa
            """Construct the prompt for the chat model.
            Args:
                messages (List[dict]): The input messages.

            Returns:
                Tuple[str, List[Image.Image], List[list]]:
                    The prompt, images, and image grid shape.
            """
            images = []
            image_grid_thws = []
            reconstructed_messages = []
            for message in messages:
                role = message['role']
                content_from_role = ''
                for item in message['content']:
                    if item['type'] == 'text':
                        content_from_role += item['text']
                    elif item['type'] == 'image':
                        image_bytes = base64.b64decode(item['image'])
                        image = Image.open(BytesIO(image_bytes)).convert('RGB')
                        image_data = image_processor(images=image)
                        pixel_values = image_data['pixel_values']
                        image_grid_thw = image_data['image_grid_thw']
                        images.extend(pixel_values)
                        image_grid_thws.append(image_grid_thw)
                        seq_len = int(image_grid_thw[0][1] *
                                      image_grid_thw[0][2] / 4)  # noqa
                        content_from_role += '<|vision_start|>' + '<|image_pad|>' * seq_len + '<|vision_end|>' + '\n'  # noqa
                reconstructed_messages.append({
                    'role': role,
                    'content': content_from_role
                })
            prompt = _self.apply_chat_template(reconstructed_messages)
            return prompt, images, image_grid_thws

        self._model.construct_prompt = _construct_prompt.__get__(self._model)

    def run(self, messages: list[dict], generation_config: dict[str, Any]):
        return self._model.chat(messages, self._tokenizer,
                                self._image_processor, generation_config)

    def ping(self):
        pass


class LeaseConnectionActorPool:

    def __init__(self, actors: list[ray.actor.ActorHandle]):
        self._actors = actors
        self._num_running_per_actor = [0] * len(actors)

    @contextlib.contextmanager
    def _load_balance(self):
        min_idx = 0
        for i in range(1, len(self._num_running_per_actor)):
            if self._num_running_per_actor[i] < self._num_running_per_actor[
                    min_idx]:
                min_idx = i
        self._num_running_per_actor[min_idx] += 1
        yield self._actors[min_idx]
        self._num_running_per_actor[min_idx] -= 1

    async def __call__(self, fn: Callable[[ray.actor.ActorHandle],
                                          ray.ObjectRef]):
        with self._load_balance() as actor:
            return await fn(actor)


def _wait_all(refs):
    ray.wait(refs, num_returns=len(refs))


class TextContext(BaseModel):
    type: Literal["text"]
    text: str


class ImageContext(BaseModel):

    class ImageURL(BaseModel):
        url: str

    type: Literal["image_url"]
    image_url: ImageURL


class Message(BaseModel):
    role: str
    content: list[TextContext | ImageContext]


class Request(BaseModel):
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 0.0
    num_beams: int = 1


class Response(BaseModel):
    text: str


class NumRunningResponse(BaseModel):
    num_running: int


class InputConverter:

    def __init__(self):
        self._http_session = aiohttp.ClientSession()
        self._base64_req_pattern = re.compile(
            r'^data:image\/(.+);base64,(.+)$')

    async def image_base64(self, url: str) -> str:
        match = self._base64_req_pattern.match(url)
        if match:
            return match.group(2)
        async with self._http_session.get(url) as response:
            return base64.b64encode(await response.read()).decode('utf-8')

    async def convert(self, request: Request) -> (list[dict], dict[str, Any]):
        messages = []
        for m in request.messages:
            context = []
            for c in m.content:
                if isinstance(c, ImageContext):
                    context.append({
                        "type":
                        "image",
                        "image":
                        await self.image_base64(c.image_url.url)
                    })
                else:
                    context.append(c.model_dump())
            messages.append({"role": m.role, "content": context})
        return messages, {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "num_beams": request.num_beams
        }


class Models:

    def __init__(self):
        num_gpus = int(math.floor(ray.available_resources()["GPU"]))
        num_model_per_gpu = 1
        model_actors = []
        num_models = num_model_per_gpu * num_gpus
        for _ in range(num_models):
            model_actors.append(
                WePOINTSModel.options(num_gpus=1 / num_model_per_gpu).remote())
        print(f"staring {num_models} models...", file=sys.stderr)
        _wait_all([actor.ping.remote() for actor in model_actors])
        print(f"{num_models} models started.", file=sys.stderr)

        self._models = LeaseConnectionActorPool(model_actors)
        self._input_converter = InputConverter()

    async def run(self, request: Request) -> Response:
        messages, generate_config = await self._input_converter.convert(request
                                                                        )
        return Response(text=await self._models(
            lambda a: a.run.remote(messages, generate_config)))


fast_api_app = fastapi.FastAPI()


@ray.serve.deployment(
    ray_actor_options={"num_cpus": 1},
    max_ongoing_requests=65536,
)
@ray.serve.ingress(app=fast_api_app)
class Application:

    def __init__(self):
        self._models = Models()
        self._num_running = 0

    @contextlib.contextmanager
    def _running_guard(self):
        self._num_running += 1
        try:
            yield
        finally:
            self._num_running -= 1

    @fast_api_app.get("/num_running", response_model=NumRunningResponse)
    async def num_running(self) -> NumRunningResponse:
        return NumRunningResponse(num_running=self._num_running)

    @fast_api_app.post("/run", response_model=Response)
    async def run(self, request: Request) -> Response:
        with self._running_guard():
            return await self._models.run(request)


def build_app(cli_args: dict[str, str]) -> ray.serve.Application:
    return Application.bind()
