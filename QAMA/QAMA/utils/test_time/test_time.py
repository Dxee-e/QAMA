import time
import torch
import yaml
import argparse
import json
import os
from icecream import ic
from QAMA.QAMultiheadAttention import QAMultiheadAttention

def test_time():
    # parser config
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='config file')
    args = parser.parse_args()
    assert args.config_path.endswith('.yaml'), 'config file must be a .yaml file'
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    solvers = config['solvers'].keys()
    time_count = {
        'forward': {solver: [] for solver in solvers},
        'backward': {solver: [] for solver in solvers},
    }
    for solver in solvers:
        model = QAMultiheadAttention(
            d_model=config['input_data']['d_model'],
            embed_dim=config['input_data']['embed_dim'],
            num_heads=config['input_data']['num_heads'],
            args_solver=config['solvers'],
            enable_solvers=solver,
        ).to(config['test']['device']) 
        
        generator = torch.Generator(device=config['test']['device'])
        generator.manual_seed(config['test']['seed'])
        
        # test time
        for _ in range(config['test']['repeat']):
            # prepare data
            Q = torch.randn(
                (config['input_data']['batch_size'], 
                 config['input_data']['seq_length'], 
                 config['input_data']['d_model']),
                dtype=torch.float32,
                generator=generator,
                device=config['test']['device'],
            )
            
            # timing
            start_time = time.time()
            results = model(Q, solver_name=solver)
            end_time = time.time()
            time_count['forward'][solver].append(end_time - start_time)
            
            loss = torch.mean(results)
            start_time = time.time()
            loss.backward()
            end_time = time.time()
            time_count['backward'][solver].append(end_time - start_time)
            
    with open(os.path.join(os.path.dirname(args.config_path), 'time_count.json'), 'w+') as f:
        json.dump(time_count, f, indent=4)
    
    time_count_forward = {solver: time_count['forward'][solver] for solver in time_count['forward'].keys()}
    time_count_backward = {solver: time_count['backward'][solver] for solver in time_count['backward'].keys()}
    print('Average time for each solver:')
    for solver in time_count['forward']:
        print(f'{solver} forward: {sum(time_count_forward[solver]) / len(time_count_forward[solver]):.4f} seconds')
        print(f'{solver} backward: {sum(time_count_backward[solver]) / len(time_count_backward[solver]):.4f} seconds')
        print(f'{solver} total: {sum(time_count_forward[solver]) + sum(time_count_backward[solver]):.4f} seconds')
    with open(os.path.join(os.path.dirname(args.config_path), 'time_count.txt'), 'w+') as f:
        f.write('Average time for each solver:\n')
        for solver in time_count['backward']:
            f.write(f'{solver} forward: {sum(time_count_forward[solver]) / len(time_count_forward[solver]):.4f} seconds\n')
            f.write(f'{solver} backward: {sum(time_count_backward[solver]) / len(time_count_backward[solver]):.4f} seconds\n')
            f.write(f'{solver} total: {sum(time_count_forward[solver]) + sum(time_count_backward[solver]):.4f} seconds\n')

if __name__=='__main__':
    test_time()