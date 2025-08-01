"""
SST-2 Dataset
The SST-2 database (https://nlp.stanford.edu/sentiment/)

download from Huggingface (https://huggingface.co/datasets/stanfordnlp/sst2)

load dataset for IMDB
"""

import torch
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from transformers import AlbertTokenizer, AlbertModel
import re
import pickle
from tqdm import tqdm

    
class SST2Dataset(Dataset):
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
        if not os.path.exists(os.path.join(self.root_path, 'raw')):
            os.makedirs(os.path.join(self.root_path, 'raw'))
            splits = {'train': 'train-00000-of-00001.parquet', 'test': 'validation-00000-of-00001.parquet'}
            base_url = 'https://huggingface.co/datasets/stanfordnlp/sst2/resolve/main/data/'
            
            for split, file_name in splits.items():
                url = f'{base_url}{file_name}?download=true'
                os.system(f'wget {url} -O {self.root_path}/raw/{split}.parquet')
        
        if not os.path.exists(os.path.join(self.root_path, 'train.csv')) or not os.path.exists(os.path.join(self.root_path, 'test.csv')):
            train_df = pd.read_parquet(os.path.join(self.root_path, 'raw', 'train.parquet'))
            test_df = pd.read_parquet(os.path.join(self.root_path, 'raw', 'test.parquet'))
            train_df.to_csv(os.path.join(self.root_path, 'train.csv'), index=False)
            test_df.to_csv(os.path.join(self.root_path, 'test.csv'), index=False)
            
    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = SST2Dataset(self.root_path, 'train')
            self.test_dataset = SST2Dataset(self.root_path, 'test')
        if stage == 'test':
            self.test_dataset = SST2Dataset(self.root_path, 'test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    