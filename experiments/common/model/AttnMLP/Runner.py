from torch import nn
from torch import optim
import torchmetrics
from .SimpleViT import SimpleViT
import torch
from icecream import ic

class Runner:
    def __init__(self, model_setting: dict, train_setting: dict, QAMA: bool = False, backend_solver: dict = None):
        self.model_setting = model_setting
        self.train_setting = train_setting
        assert QAMA is True, "QAMA must be True to set different attention module."
        self.QAMA = QAMA
        
        model_setting['attn_name'] = backend_solver['solver']
        self.model = SimpleViT(**model_setting).to(train_setting['device'])
        self.loss_fn = nn.CrossEntropyLoss() 
        
    
    def forward(self, x):
        return self.model(x, solver_name=self.solver_name)
    
    def metrics_accuracy(self, logits, labels):
        # logits: [b, n_classes]
        # labels: [b]
        return (logits.argmax(dim=1) == labels).float().mean()
    
    def training_step(self, batch):
        x, y = batch
        x = x.to(self.train_setting['device'])
        y = y.to(self.train_setting['device'])
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = self.metrics_accuracy(logits, y)
        return {'loss': loss, 'acc': acc}
    
    def validation_step(self, batch):
        x, y = batch
        x = x.to(self.train_setting['device'])
        y = y.to(self.train_setting['device'])
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = self.metrics_accuracy(logits, y)
        return {'loss': loss, 'acc': acc}

