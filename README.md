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

**2024.10.05** We open-sourced the inference code of [POINTS](https://huggingface.co/WePOINTS/POINTS-Yi-1-5-9B-Chat)🔥🔥🔥.
<br>
**2024.09.07** We released the paper about the first vision-language model, [POINTS](https://arxiv.org/abs/2409.04828)🚀🚀🚀.
<br>
**2024.05.20** We released the [paper](https://arxiv.org/abs/2405.11850) to reveal some overlooked aspects in vision-language models🚀🚀🚀.

## Model Zoo

|         Model         |    Date    |                                        Download                                         |                     Note                     |
| :-------------------: | :--------: | :-------------------------------------------------------------------------------------: | :------------------------------------------: |
| POINTS-Yi-1.5-9B-Chat | 2024.10.03 | 🤗 [HF link](https://huggingface.co/WePOINTS/POINTS-Yi-1-5-9B-Chat)<br>🤖 [MS link](<>) | Strong performance with affordable stategies |

## Installation

```sh
git clone https://github.com/WePOINTS/WePOINTS.git
cd WePOINTS
pip install -e .
```

## How to Use?

We provide the usage of POINTS, using huggingface 🤗 transformers library.
<br>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import CLIPImageProcessor
from PIL import Image
import torch
import requests
from io import BytesIO


image_url = 'https://github.com/user-attachments/assets/83258e94-5d61-48ef-a87f-80dd9d895524'
response = requests.get(image_url)
image_data = BytesIO(response.content)
pil_image = Image.open(image_data)
prompt = 'please describe the image in detail'
model_path = 'WePOINTS/POINTS-Yi-1-5-9B-Chat'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, device_map='cuda').to(torch.bfloat16)
image_processor = CLIPImageProcessor.from_pretrained(model_path)
generation_config = {
    'max_new_tokens': 1024,
    'temperature': 0.0,
    'top_p': 0.0,
    'num_beams': 1,
}
res = model.chat(
    pil_image,
    prompt,
    tokenizer,
    image_processor,
    True,
    generation_config
)
print(res)
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

## Evaluation

|     Benchmark      | InternVL2-8B | LLaVA-OneVision | POINTS |
| :----------------: | :----------: | :-------------: | :----: |
|   MMBench-dev-en   |      -       |      80.8       |  83.2  |
|     MathVista      |     58.3     |      62.3       |  60.7  |
| HallucinationBench |     45.0     |      31.6       |  48.0  |
|      OCRBench      |     79.4     |      62.2       |  70.6  |
|        AI2D        |     83.6     |      82.4       |  78.5  |
|       MMVet        |     54.3     |      51.9       |  50.0  |
|       MMStar       |     61.5     |      61.9       |  56.4  |
|        MMMU        |     51.2     |      47.9       |  46.9  |
|     ScienceQA      |     97.1     |      95.4       |  92.9  |
|        MME         |    2215.1    |     1993.6      | 2017.8 |
|    RealWorldQA     |     64.2     |      69.9       |  65.9  |
|     LLaVA-Wild     |     73.3     |      81.0       |  69.3  |

## Citation

If you find our work helpful, feel free to cite us:

```
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
