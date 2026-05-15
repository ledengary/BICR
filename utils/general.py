import os
import json
import jsonlines
import random
import numpy as np
import torch
import re
from transformers import AutoTokenizer

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def set_visible_cudas(gpu_ids):
    print(f"Visible CUDAs before setting: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    print(f"Visible CUDAs after setting: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

def get_uncertainty_query():
    return (
        "Is the proposed answer correct?\n"
        "A) no\nB) yes\n"
        "Reply with A or B only.\n"
        "Answer: "
    )