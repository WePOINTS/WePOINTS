import json
from typing import List

import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def read_jsonl_file(file_path: str) -> List[any]:
    """Read a JSONL file and return a list of objects.

    Args:
        file_path (str): The path of the JSONL file to be read.

    Returns:
        List[any]: A list of objects read from the JSONL file.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        objects = [json.loads(line.strip()) for line in lines]
    return objects


def dump_list_to_jsonl_file(file_path: str, data: List[any]) -> None:
    """Dump a list of objects to a JSONL file.

    Args:
        file_path (str): The path of the JSONL file to be written.
        data (List[any]): A list of objects to be written to the JSONL file.

    Returns:
        None: None.
    """

    with open(file_path, 'w') as f:
        for item in data:
            json_str = json.dumps(item)
            f.write(json_str + '\n')

    return None


class Qwen2VL:
    """Qwen2VL model wrapper.

    This class is used to get the perplexity of the generated caption
    for each image.

    Args:
        path (str): The path of the model checkpoint.
    """
    def __init__(self, path: str) -> None:
        print(f"load model from '{path}'")
        # load model from checkpoint
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map='auto',
            low_cpu_mem_usage=True)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(path)

    def _get_model_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(**inputs, return_dict=False)[0].detach().cpu()
        return outputs

    def _calc_ppl(self, output_prob: list) -> float:
        output_prob = torch.tensor(output_prob)
        ppl = torch.exp(-torch.mean(torch.log(output_prob)))
        return ppl.item()

    def get_output_info(self, standard_jsonl_line: dict, idx: int) -> dict:
        """Get the perplexity of the generated caption for each image.

        Args:
            standard_jsonl_line (dict): A standard jsonl line.
            idx (int): The index of the standard jsonl line.

        Returns:
            dict: A jsonl line with the perplexity of the generated
                caption for each image.
        """
        base64_image = standard_jsonl_line['image']
        caption = standard_jsonl_line['caption']

        prompt = 'please generate a caption for this image.'
        messages_prompt = [{
            'role':
            'user',
            'content': [
                {
                    'type': 'image',
                    'image': f'data:image;base64,{base64_image}'
                },
                {
                    'type': 'text',
                    'text': prompt
                },
            ],
        }]
        text_prompt = self.processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True)
        image_inputs_prompt, video_inputs_prompt = process_vision_info(
            messages_prompt)
        inputs_prompt = self.processor(
            text=[text_prompt],
            images=image_inputs_prompt,
            videos=video_inputs_prompt,
            padding=True,
            return_tensors='pt',
        )

        messages_caption = [{
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': caption
                },
            ],
        }]
        text_caption = self.processor.apply_chat_template(
            messages_caption, tokenize=False, add_generation_prompt=False)
        inputs_caption = self.processor(
            text=[text_caption],
            padding=True,
            return_tensors='pt',
        )

        valid_len = inputs_caption['input_ids'].shape[1] - 14

        inputs_prompt['input_ids'] = torch.cat(
            [inputs_prompt['input_ids'], inputs_caption['input_ids'][:, 14:]],
            dim=-1)
        inputs_prompt['attention_mask'] = torch.cat([
            inputs_prompt['attention_mask'],
            inputs_caption['attention_mask'][:, 14:]
        ],
                                                    dim=-1)
        inputs = inputs_prompt.to('cuda')

        # get all logits
        with torch.no_grad():
            all_logits = self._get_model_logits(inputs)
            output_logits = all_logits[:, -valid_len - 1:-2, :].squeeze(dim=0)
            input_labels = inputs['input_ids'].detach().cpu(
            )[:, -valid_len:-1].squeeze(dim=0)
            output_prob = F.softmax(output_logits.float(), dim=-1)

        ppl_jsonl_line = {
            'idx':
            idx,
            'ppl':
            self._calc_ppl(
                torch.gather(output_prob,
                             dim=-1,
                             index=input_labels.unsqueeze(dim=-1)).squeeze(
                                 dim=-1).to(torch.bfloat16).detach().tolist()),
        }

        return ppl_jsonl_line


class PPLPipeline:
    """Pipeline to filter the pre-training data by the perplexity of the
    generated caption.

    Args:
        model_name (str): The name of the model.
        model_path (str): The path of the model checkpoint.
        original_file_path (str): The path of the original jsonl file.
        filtered_file_path (str): The path of the filtered jsonl file.
    """
    def __init__(self, model_name: str, model_path: str,
                 original_file_path: str, filtered_file_path: str) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.original_file_path = original_file_path
        self.filtered_file_path = filtered_file_path

    def _init_runner(self) -> None:
        if self.model_name == 'Qwen2VL':
            self.runner = Qwen2VL(self.model_path)
        else:
            raise NotImplementedError

    def gen_ppl(self, jsonl_lines: List[dict]) -> List[dict]:
        """Generate perplexity for each image caption.

        Args:
            jsonl_lines (List[dict]): A list of standard jsonl lines.

        Returns:
            List[dict]: A list of jsonl lines with the perplexity of the
                generated caption for each image.
        """
        self._init_runner()
        ppl_jsonl_lines = []
        for idx, jsonl_line in enumerate(tqdm(jsonl_lines)):
            try:
                ppl_jsonl_line = self.runner.get_output_info(jsonl_line, idx)
                ppl_jsonl_lines.append(ppl_jsonl_line)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('CUDA out of memory, skip this sample')
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            except KeyboardInterrupt:
                exit(1)
            except Exception as e:
                print(e)
        return ppl_jsonl_lines

    def filter(self, remaining_ratio: float = 0.2) -> None:
        """Filter the original jsonl file by the perplexity of the generated
        caption.

        Args:
            remaining_ratio (float, optional): The ratio of the remaining data.
                Defaults to 0.2.
        """
        jsonl_lines = read_jsonl_file(self.original_file_path)
        ppl_lines = self.gen_ppl(jsonl_lines)
        sorted_ppl_lines = sorted(ppl_lines, key=lambda x: x['ppl'])
        idx_list = [d['idx'] for d in sorted_ppl_lines]
        select_idx_list = idx_list[:int(len(idx_list) * remaining_ratio)]
        out_lines = [jsonl_lines[idx] for idx in select_idx_list]
        dump_list_to_jsonl_file(self.filtered_file_path, out_lines)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen2VL')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--original_file_path', type=str, default='')
    parser.add_argument('--filtered_file_path', type=str, default='')
    args = parser.parse_args()

    pipe_runner = PPLPipeline(model_name=args.model_name,
                              model_path=args.model_path,
                              original_file_path=args.original_file_path,
                              filtered_file_path=args.filtered_file_path)
    pipe_runner.filter()
