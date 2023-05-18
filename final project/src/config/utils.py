import os
from collections import OrderedDict
from typing import List, Tuple, Union

import torch
import random
import numpy as np
from path import Path
from torch.utils.data import DataLoader
import math 
from torch.nn.functional import normalize
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()
OUT_DIR = _PROJECT_DIR / "out"
TEMP_DIR = _PROJECT_DIR / "temp"


def fix_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clone_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]
) -> OrderedDict[str, torch.Tensor]:
    if isinstance(src, OrderedDict):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.items()
            }
        )
    if isinstance(src, torch.nn.Module):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.state_dict(keep_vars=True).items()
            }
        )


def trainable_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module], requires_name=False
) -> Union[List[torch.Tensor], Tuple[List[str], List[torch.Tensor]]]:
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(param)
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(param)
                keys.append(name)

    if requires_name:
        return keys, parameters
    else:
        return parameters

def softmax_criterion(outs, target):
	loss_fn = nn.CrossEntropyLoss()
	outs = softmax(outs)
	# loss = softmax_cross_entropy(outs, target)
	loss = loss_fn(outs, target)
	return loss

def softmax(x):
    exp_x = torch.exp(x - torch.max(x))
    return exp_x / exp_x.sum()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    # criterion=softmax_criterion(), 
    device=torch.device("cpu"),
) -> Tuple[float, float, int]:
    model.eval()
    correct = 0
    loss = 0
    sample_num = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        logits = softmax(logits)
        loss += criterion(logits, y).item()
        pred = torch.argmax(logits, -1)
        correct += (pred == y).sum().item()
        sample_num += len(y)
    return loss, correct, sample_num


def multiply_model(model1, model2, normalize=False):
    params1 = torch.cat([torch.flatten(p) for p in model1])
    params2 = torch.cat([torch.flatten(p) for p in model2])
    cos_sim = F.cosine_similarity(params1.reshape(1, -1), params2.reshape(1, -1))
    if normalize==True:
        return cos_sim.item(), torch.linalg.norm(params1.reshape(1, -1)).item(), torch.linalg.norm(params2.reshape(1, -1)).item()
    else:
        return cos_sim.item()

def num_layers(model):
    nlayers = 0
    for name, _ in model.named_parameters():
        nlayers += 1
    return nlayers

