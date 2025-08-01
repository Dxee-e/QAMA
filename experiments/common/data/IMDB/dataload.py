"""
IMDB Dataset
The IMDB database (https://ai.stanford.edu/~amaas/data/sentiment/)

load dataset for IMDB
"""

import torch
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
import re
import pickle
from tqdm import tqdm


class IMDBDataset(Dataset):
    def __init__(self, root_path, flag):
        self.df = pd.read_csv(os.path.join(root_path, f'{flag}.csv'))
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sentence = row['sentence']
        label = row['label']
        return sentence, torch.tensor(label, dtype=torch.long)
    

class Dataset:
    def __init__(self, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_path = os.path.abspath(os.path.dirname(__file__))

    def perpare_data(self):
        if not os.path.exists(os.path.join(self.root_path, 'aclImdb')):
            url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
            os.system(f'wget {url} -P {self.root_path}')
            os.system(f'tar -xzf {os.path.join(self.root_path, "aclImdb_v1.tar.gz")} -C {self.root_path}')
            os.remove(os.path.join(self.root_path, 'aclImdb_v1.tar.gz'))        
        if not os.path.exists(os.path.join(self.root_path, 'train.csv')) \
                or not os.path.exists(os.path.join(self.root_path, 'test.csv')):
            
            def get_data(path, flag):
                pos_files = os.listdir(os.path.join(path, 'pos'))
                neg_files = os.listdir(os.path.join(path, 'neg'))
                
                pos_all = []
                neg_all = []
                for pf, nf in tqdm(zip(pos_files, neg_files), total=len(pos_files)):
                    with open(os.path.join(path, 'pos', pf), 'r') as f:
                        sentence = f.read()
                        pos_all.append(sentence)
                    with open(os.path.join(path, 'neg', nf), 'r') as f:
                        sentence = f.read()
                        neg_all.append(sentence)
                x_original = pos_all + neg_all
                y_original = [1] * len(pos_all) + [0] * len(neg_all)
                
                df = {
                    'idx': list(range(len(x_original))),
                    'sentence': x_original,
                    'label': y_original,
                }
                df = pd.DataFrame(df)
                df.to_csv(os.path.join(self.root_path, f'{flag}.csv'), index=False)
        
            get_data(os.path.join(self.root_path, 'aclImdb', 'train'), 'train')
            get_data(os.path.join(self.root_path, 'aclImdb', 'test'), 'test')
            
    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = IMDBDataset(self.root_path, 'train')
            self.test_dataset = IMDBDataset(self.root_path, 'test')
        if stage == 'test':
            self.test_dataset = IMDBDataset(self.root_path, 'test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    