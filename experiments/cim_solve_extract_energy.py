"""
only server for cim_run
code is designed not in general
"""
import yaml
import argparse
from icecream import ic
from importlib import import_module
import torch
import os
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd

batch_idx = 9668


def load_class(module_path: str, class_name: str):
    module = import_module(module_path)
    class_ = getattr(module, class_name)
    return class_

torch.manual_seed(42)

# dataset
dataload_path = 'common.data.IMDB.dataload'
data_module = load_class(dataload_path, 'Dataset')(batch_size=1)
# ic(data_module)
data_module.perpare_data()
data_module.setup('test')
test_dataset = data_module.test_dataset

# config
config = yaml.safe_load(open('./results/IMDB-TextTransformer-QAMA-c_sa/config.yaml', 'r'))
config['backend_solver']['solver'] = 'cim'
del config['backend_solver']['c_sa']
config['backend_solver']['cim'] = {
    # kaiwu SDK init license
    "user_id": '69878024601862146',
    "sdk_code": '0i4T6LY1XygfwN3MWa8Fjq27OaT0sq',
    # stage
    'stage': 'solve', # 'generate' or 'solve'
    'batch_idx': [batch_idx],
}

# runner
runner_path = 'common.model.TextTransformer.Runner'
Runner = load_class(runner_path, 'Runner')
runner = Runner(
    model_setting=config['model_setting'], 
    train_setting=config['train_setting'], 
    QAMA=True,
    backend_solver=config['backend_solver'],
)
runner.model.load_state_dict(torch.load(os.path.join('./results/IMDB-TextTransformer-QAMA-c_sa', 'saved_model', f'epoch_50.pth'), weights_only=True))
runner.model.eval()

sentences, labels = [], []
batch = test_dataset[batch_idx]
sentences.append(batch[0])
labels.append(batch[1])

with torch.no_grad():
    logits = runner.model.forward(sentences, solver_name='cim')
logits = logits.cpu().numpy()
print(f"logits: {logits}, labels: {labels}")
print(f"accuracy: {(logits.argmax(axis=1) == labels).mean()}")

