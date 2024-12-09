# WePOINTS

<p align="center">
    <img src="https://github.com/user-attachments/assets/4d5424e0-af7e-4a5e-8c77-6743e21f79db" width="700"/>
<p>
<p align="center">
        🤗 <a href="https://huggingface.co/WePOINTS/POINTS-Yi-1-5-9B-Chat">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="">Blog</a> &nbsp&nbsp| &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2409.04828">Paper</a> &nbsp&nbsp  </a>
</p>

## Introduction

We foresee a future where content understanding and generation are seamlessly unified within a single model. To this end, we have launched the WePOINTS project. WePOINTS is a suite of multimodal models designed to create a unified framework that accommodates various modalities. These models are being developed by researchers at WeChat AI, leveraging the latest advancements and cutting-edge techniques in multimodal models.


## What's New?

**2024.12.06** We are releasing POINTS1.5, a significantly enhanced model compared to POINTS, now with bilingual support. Notably, POINTS1.5-7B ranks first on the OpenCompass leaderboard among all models under 10B. The accompanying paper will be available soon.🔥🔥🔥.
<br>
**2024.11.02** Add the [demo script](scripts/pretrain_filtering_with_ppl.py) to filter the pre-training data by perplexity.
<br>
**2024.10.15** We released POINTS with Qwen2.5-7B 🔥🔥🔥.
<br>
**2024.10.05** We open-sourced the inference code of [POINTS](https://huggingface.co/WePOINTS/POINTS-Yi-1-5-9B-Chat)🔥🔥🔥.
<br>
**2024.09.07** We released the paper about the first vision-language model, [POINTS](https://arxiv.org/abs/2409.04828)🚀🚀🚀.
<br>
**2024.05.20** We released the [paper](https://arxiv.org/abs/2405.11850) to reveal some overlooked aspects in vision-language models🚀🚀🚀.


## Release Plan

**POINTS-Yi-1-5-9B-Chat re-opens now 🎉🎉🎉**

- The technical report for POINTS1.5 will be released soon.
- We will also be releasing the pre-training dataset for POINTS1.5.
- Additionally, POINTS1.5 will be integrated into [SGLang](https://github.com/sgl-project/sglang).


## Model Zoo

|          Model          |    Date    |                                         Download                                          |                     Note                     |
| :---------------------: | :--------: | :---------------------------------------------------------------------------------------: | :------------------------------------------: |
| POINTS-1-5-Qwen-2-5-7B-Chat | 2024.12.06 | 🤗 [HF link](https://huggingface.co/WePOINTS/POINTS-1-5-Qwen-2-5-7B-Chat)<br>🤖 [MS link](<>) |                  Qwen2.5-7B                  |
| POINTS-Qwen-2-5-7B-Chat | 2024.10.15 | 🤗 [HF link](https://huggingface.co/WePOINTS/POINTS-Qwen-2-5-7B-Chat)<br>🤖 [MS link](<>) |                  Qwen2.5-7B                  |
|  POINTS-Yi-1.5-9B-Chat  | 2024.10.03 |  🤗 [HF link](https://huggingface.co/WePOINTS/POINTS-Yi-1-5-9B-Chat)<br>🤖 [MS link](<>)  | Strong performance with affordable stategies |

## Installation

```sh
git clone https://github.com/WePOINTS/WePOINTS.git
cd WePOINTS
pip install -e .
```

## How to Use?

We provide the usage of POINTS1.5, using Hugging Face 🤗 transformers library.
<br>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from wepoints.utils.images import Qwen2ImageProcessorForPOINTSV15
import torch
from PIL import Image
import requests
from io import BytesIO


model_path = 'WePOINTS/POINTS-1-5-Qwen-2-5-7B-Chat'
model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.float16,
                                                 device_map='cuda') 
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
image_processor = Qwen2ImageProcessorForPOINTSV15.from_pretrained(model_path)


image_url = 'https://github.com/user-attachments/assets/83258e94-5d61-48ef-a87f-80dd9d895524'
response = requests.get(image_url)
image_data = BytesIO(response.content)
pil_image = Image.open(image_data)
pil_image = pil_image.save('image.jpg')
prompt = 'please describe the image in detail'

content = [
        dict(type='image', image='image.jpg'),
        dict(type='text', text=prompt)
    ]
messages = [
        {
            'role': 'user',
            'content': content
        }
    ]
generation_config = {
        'max_new_tokens': 1024,
        'temperature': 0.0,
        'top_p': 0.0,
        'num_beams': 1,
    }
response = model.chat(
    messages,
    tokenizer,
    image_processor,
    generation_config
)
print(response)
```

## How to Evaluate?

We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to evaluate the performance of our models. Please follow the [installation instructions](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/get_started/Quickstart.md) to install the toolkit. Then, you can evaluate POINTS using the following command:

```sh
# before running the evaluation below, please enter the root directory of VLMEvalKit
torchrun --nproc-per-node=8 --master_port=8888 --no-python python run.py --data MMMU_DEV_VAL MMBench_DEV_EN MMBench_TEST_EN_V11 MMBench_TEST_CN_V11 HallusionBench OCRBench AI2D_TEST MMStar MMVet MathVista_MINI MME RealWorldQA LLaVABench POPE  --model POINTS-Yi-1.5-9B-Chat --verbose --work-dir ./
```

## Model Soup

POINTS is the pioneering work that proposes applying model soup to models fine-tuned with different instruction datasets. This approach combines the benefits of these models and enhances the performance of the final averaged model. The following code snippet demonstrates how to use model soup to combine the models:

```python
from wepoints.utils import model_soup

# models fine-tuned with different instruction datasets
model_paths_or_names = [
  'first_model_path',
  'second_model_path',
  'third_model_path'
]
save_path = '/path/to/save/model'
model_soup(model_paths_or_names, save_path)
```

## CATTY

CATTY is a brand new strategy to split a large-resolution image into small patches of the same size. Compared to previsou approaches, CATTY can preserve the original image aspect ratio.

```python
from PIL import Image
from wepoints.utils.images.catty import split_image_with_catty

image_path = '/path/to/local/image.jpg'
# used to save the split images for debugging
save_folder = '/path/to/save/folder'
image = Image.open(image_path)
sub_images = split_image_with_catty(image, save_folder=save_folder, do_resize=True)
```

## Filter the Pre-training Data by Perplexity

We provide a script to filter the pre-training data by perplexity using Qwen2VL. Please first the download the demo pre-training data [here](https://huggingface.co/datasets/WePOINTS/POINTS-PT-PPL-DEMO). And then run the following command:

```sh
python scripts/pretrain_filtering_with_ppl.py --model_name Qwen2VL --model_path /path/to/model --original_file_path data.jsonl --filtered_file_path filtered_data.jsonl
```

## Evaluation Results

| Benchmark | Qwen2-VL-7B | POINTS-7B | POINTS1.5-7B |
| :-------: | :----------: | :-------------: | :----: |
| MMBench-TEST-avg      | 81.0 | 78.0 | 80.7 |
| MMStar                | 60.7 | 60.9 | 61.1 |
| MMMU                  | 53.7 | 51.4 | 53.8 |
| MathVista             | 61.4 | 63.0 | 66.4 |
| HallucinationBench    | 50.4 | 45.6 | 50.0 |
| AI2D                  | 83.0 | 81.2 | 81.4 |
| OCRBench              | 84.3 | 71.7 | 82.3 | 
| MMVet                 | 61.8 | 47.9 | 62.2 |
| Average               | 67.0 | 62.5 | 67.4 |


## Acknowledgements

POINTS1.5 adapts the NaViT from Qwen2-VL, and we extend our gratitude to the Qwen Team for their outstanding work. We are also grateful to [Haodong Duan](https://scholar.google.com.hk/citations?user=vi3W-m8AAAAJ&hl=zh-CN) for his assistance in evaluating POINTS1.5.


## Citation

If you find our work helpful, feel free to cite us:

```
@article{points1.5,
  title={POINTS1.5: Building a Vision-Language Model towards Real World Applications},
  author={Liu, Yuan and Le Tian and Xiao Zhou and Xinyu Gao and Kavio Yu and Yang Yu and Jie Zhou},
  journal={Coming soon},
  year={2024}
}

@article{liu2024points,
  title={POINTS: Improving Your Vision-language Model with Affordable Strategies},
  author={Liu, Yuan and Zhao, Zhongyin and Zhuang, Ziyuan and Tian, Le and Zhou, Xiao and Zhou, Jie},
  journal={arXiv preprint arXiv:2409.04828},
  year={2024}
}

@article{liu2024rethinking,
  title={Rethinking Overlooked Aspects in Vision-Language Models},
  author={Liu, Yuan and Tian, Le and Zhou, Xiao and Zhou, Jie},
  journal={arXiv preprint arXiv:2405.11850},
  year={2024}
}
```
