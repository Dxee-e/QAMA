"""
Fashion-MNIST is a dataset of Zalando's article images.

Introduced by: Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747
(https://github.com/zalandoresearch/fashion-mnist)

download from torchvision.datasets.Fashion-MNIST
load dataset for Fashion-MNIST

x.shape [1, 28, 28]
y.shape [1]
"""

import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from typing import Literal
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
        torchvision.datasets.FashionMNIST(root=self.root_path, train=True, download=True)
        torchvision.datasets.FashionMNIST(root=self.root_path, train=False, download=True)
        
    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = torchvision.datasets.FashionMNIST(
                root=self.root_path, train=True, download=False, transform=self.transform
            )
            self.test_dataset = torchvision.datasets.FashionMNIST(
                root=self.root_path, train=False, download=False, transform=self.transform
            )
        if stage == 'test':
            self.test_dataset = torchvision.datasets.FashionMNIST(
                root=self.root_path, train=False, download=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
