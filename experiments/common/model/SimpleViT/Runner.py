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
        self.QAMA = QAMA
    
        if self.QAMA:
            self.solver_name = backend_solver['solver']
            self.solver_args = {self.solver_name: backend_solver[self.solver_name]}
        else:
            self.solver_name = None
            self.solver_args = None
        
        self.model = SimpleViT(**model_setting).to(train_setting['device'])
        if self.QAMA:
            self.replace_module()
        self.loss_fn = nn.CrossEntropyLoss() 
        
    def replace_module(self):
        from QAMA.QAMultiheadAttention import QAMultiheadAttention
        
        def replace_attention(module):
            for name, child in module.named_children():
                if child.__class__.__name__ == 'Attention':
                    d_model = child.dim
                    embed_dim = child.dim_head
                    num_heads = child.heads
                    new_attention = QAMultiheadAttention(
                        d_model=d_model,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        args_solver=self.solver_args,
                        enable_solvers=self.solver_name,
                    ).to(self.train_setting['device'])
                    setattr(module, name, new_attention)
                replace_attention(child)
        
        replace_attention(self.model)
    
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
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = self.metrics_accuracy(logits, y)
        return {'loss': loss, 'acc': acc}
    
    def validation_step(self, batch):
        x, y = batch
        x = x.to(self.train_setting['device'])
        y = y.to(self.train_setting['device'])
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = self.metrics_accuracy(logits, y)
        return {'loss': loss, 'acc': acc}

