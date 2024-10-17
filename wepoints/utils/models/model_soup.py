from collections import OrderedDict
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def model_soup(model_path_or_names: List[str],
               save_path: str,
               trust_remote_code: bool = True,
               dtype: torch.dtype = torch.bfloat16) -> None:
    """Average the weights of multiple models and save the averaged model.

    Args:
        model_path_or_names (List[str]): A list of model paths or huggingface
            model names.
        save_path (str): The path to save the averaged model.
        trust_remote_code (bool, optional): Whether to trust the remote code
            when loading the models. Defaults to True.
        dtype (torch.dtype, optional): The dtype of the averaged model.
            Defaults to torch.bfloat16.

    Returns:
        None
    """
    state_dicts = []
    for model_path_or_name in model_path_or_names:
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_name, trust_remote_code=trust_remote_code).to(dtype)
        state_dict = model.state_dict()
        state_dicts.append(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path_or_names[0], trust_remote_code=trust_remote_code)  # noqa
    weight_keys = list(state_dicts[0].keys())
    average_state_dict = OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for state_dict in state_dicts:
            key_sum += state_dict[key]
        average_state_dict[key] = key_sum / len(state_dicts)
    model.load_state_dict(average_state_dict, strict=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f'Successfully saved averaged model to {save_path}')
