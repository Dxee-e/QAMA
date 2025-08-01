import yaml
import argparse
from icecream import ic
from importlib import import_module
import torch
import os
import shutil
from tqdm import tqdm
import pandas as pd


def paser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_path", type=str, help="Path to the config file")
    parser.add_argument("--epoch", type=int, default=-1, help="Epoch to load the model")
    args = parser.parse_args()
    return args

def load_config(config_path: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_class(module_path: str, class_name: str):
    module = import_module(module_path)
    class_ = getattr(module, class_name)
    return class_


def infer():
    args = paser_args()
    config_path = os.path.join(args.result_path, 'config.yaml')
    config = load_config(config_path)
    ic(config)
    
    torch.manual_seed(config['train_setting']['seed'])
    
    enable_QAMA = config['base']['QAMA']
    
    # runner
    runner_path = 'common.model.' + config['base']['model'] + '.Runner'
    Runner = load_class(runner_path, 'Runner')
    runner = Runner(
        model_setting=config['model_setting'], 
        train_setting=config['train_setting'], 
        QAMA=enable_QAMA,
        backend_solver=config['backend_solver'] if enable_QAMA else None,
    )
    
    # model summary
    total_params = sum(p.numel() for p in runner.model.parameters())
    trainable_params = sum(p.numel() for p in runner.model.parameters() if p.requires_grad)
    ic(f"Total parameters: {total_params}")
    ic(f"Trainable parameters: {trainable_params}")
    

    # dataset
    dataload_path = 'common.data.' + config['base']['dataset'] + '.dataload'
    data_module = load_class(dataload_path, 'Dataset')(batch_size=config['train_setting']['batch_size'])
    # ic(data_module)
    data_module.perpare_data()
    data_module.setup('test')
    test_dataloader = data_module.test_dataloader()
    
    
    # load model
    epoch = args.epoch if args.epoch != -1 else config['train_setting']['num_epochs']
    runner.model.to(config['train_setting']['device'])
    runner.model.load_state_dict(torch.load(os.path.join(args.result_path, 'saved_model', f'epoch_{epoch}.pth'), weights_only=True, map_location=config['train_setting']['device']))
    
    # output
    output_results = {
        'batch_idx': [],
        'label': [],
        'pred': [],
        'correct': [],
        'sentence': [],
    }
    for i in range(config['model_setting']['num_classes']):
        output_results[f'logits_{i}'] = []
    
    # test
    runner.model.eval()
    with torch.no_grad():
        cnt = 0
        for batch in tqdm(test_dataloader, desc="Test Batches", leave=False, total=len(test_dataloader)):
            sentences, labels = batch
            if type(sentences) is torch.Tensor:
                sentences = sentences.to(config['train_setting']['device'])
            output = runner.forward(sentences).detach().cpu()
            pred = output.argmax(dim=1)
            
            batch_size = pred.shape[0]
            for i in range(batch_size):
                output_results['batch_idx'].append(cnt)
                output_results['label'].append(labels[i].item())
                output_results['sentence'].append(sentences[i])
                output_results['pred'].append(pred[i].item())
                output_results['correct'].append(pred[i].item() == labels[i].item())
                for j in range(config['model_setting']['num_classes']):
                    output_results[f'logits_{j}'].append(output[i, j].item())

                cnt += 1
                
    # save results
    output_path = os.path.join(args.result_path, f"test_output_{epoch}.csv")
    pd.DataFrame(output_results).to_csv(output_path, index=False)
            

if __name__ == "__main__":
    infer()
