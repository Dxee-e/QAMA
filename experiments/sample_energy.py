"""
only server for sample_energy
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

batch_idx = 2201


def load_class(module_path: str, class_name: str):
    module = import_module(module_path)
    class_ = getattr(module, class_name)
    return class_

torch.manual_seed(42)

# dataset
dataload_path = 'common.data.CIFAR10.dataload'
data_module = load_class(dataload_path, 'Dataset')(batch_size=1)
# ic(data_module)
data_module.perpare_data()
data_module.setup('test')
test_dataset = data_module.test_dataset

# config
config = yaml.safe_load(open('./results/CIFAR10-SimpleViT-QAMA-c_sa/config.yaml', 'r'))

# runner
runner_path = 'common.model.SimpleViT.Runner'
Runner = load_class(runner_path, 'Runner')
runner = Runner(
    model_setting=config['model_setting'], 
    train_setting=config['train_setting'], 
    QAMA=True,
    backend_solver=config['backend_solver'],
)
runner.model.load_state_dict(torch.load(os.path.join('./results/CIFAR10-SimpleViT-QAMA-c_sa', 'saved_model', f'epoch_50.pth'), weights_only=True))
runner.model.eval()

img, label = test_dataset[batch_idx]
torch.save(img, './sample_energy/img.pt')

output_results = {
    'batch_idx': [],
    'label': [],
    'pred': [],
    'correct': [],
}
with torch.no_grad():
    logits = runner.model.forward(img.unsqueeze(0).to('cuda'), solver_name='c_sa').squeeze(0)
    logits = logits.cpu().numpy()
    
    output_results['batch_idx'].append(batch_idx)
    output_results['label'].append(label)
    output_results['pred'].append(logits.argmax().item())
    output_results['correct'].append(logits.argmax().item() == label)
    
    for j in range(config['model_setting']['num_classes']):
        if f'logits_{j}' not in output_results:
            output_results[f'logits_{j}'] = []
        output_results[f'logits_{j}'].append(logits[j].item())

pd.DataFrame(output_results).to_csv(os.path.join('./sample_energy', 'c_sa_output_results.csv'), index=False)