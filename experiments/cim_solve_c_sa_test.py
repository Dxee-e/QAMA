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

select_batch_idx = np.load('./cim_run/random_indices.npy')


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
for batch_idx in select_batch_idx:
    batch = test_dataset[batch_idx]
    sentences.append(batch[0])
    labels.append(batch[1])

output_results = {
    'batch_idx': [],
    'label': [],
    'pred': [],
    'correct': [],
    'sentence': [],
}
with torch.no_grad():
    logits = runner.model.forward(sentences, solver_name='c_sa')
    logits = logits.cpu().numpy()
    
    for i, batch_idx in enumerate(select_batch_idx):
        output_results['batch_idx'].append(batch_idx)
        output_results['label'].append(labels[i].item())
        output_results['pred'].append(logits[i].argmax().item())
        output_results['correct'].append(logits[i].argmax().item() == labels[i].item())
        output_results['sentence'].append(sentences[i])
        
        for j in range(config['model_setting']['num_classes']):
            if f'logits_{j}' not in output_results:
                output_results[f'logits_{j}'] = []
            output_results[f'logits_{j}'].append(logits[i][j].item())

pd.DataFrame(output_results).to_csv(os.path.join('./cim_run', 'c_sa_output_results.csv'), index=False)