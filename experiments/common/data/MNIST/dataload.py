"""
MNIST Dataset
The MNIST database of handwritten digits (http://yann.lecun.com)

download from torchvision.datasets.MNIST

load dataset for MNIST
"""

import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import os
from torch.utils.data import DataLoader

class Dataset:
    def __init__(self, batch_size: int = 32, num_workers: int = 2):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))]
        )
        self.root_path = os.path.abspath(os.path.dirname(__file__))

    def perpare_data(self):
        torchvision.datasets.MNIST(root=self.root_path, train=True, download=True)
        torchvision.datasets.MNIST(root=self.root_path, train=False, download=True)
        
    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = torchvision.datasets.MNIST(
                root=self.root_path, train=True, download=False, transform=self.transform
            )
            self.test_dataset = torchvision.datasets.MNIST(
                root=self.root_path, train=False, download=False, transform=self.transform
            )
        if stage == 'test':
            self.test_dataset = torchvision.datasets.MNIST(
                root=self.root_path, train=False, download=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
