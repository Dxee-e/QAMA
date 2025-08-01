import yaml
import argparse
from icecream import ic
from importlib import import_module
import torch
import os
import shutil
from tqdm import tqdm
import logging
import json

class Trainer:
    def __init__(self):
        # init args and config
        self.args = self.paser_args()
        self.config = self.load_config(self.args.config_path)
        self.enable_QAMA = self.config['base']['QAMA']
        self.root_dir = self.init_root_dir()
        self.logger = self.init_logger(self.root_dir)
        
        # log config
        self.logger.info(f"Configuration: {self.config}")
        
        # set seed 
        torch.manual_seed(self.config['train_setting']['seed'])
        
        # load runner
        self.runner = self.load_runner()
        self.model_summary()
        
        # load dataset
        self.train_dataloader, self.val_dataloader = self.load_dataset()
        self.dataset_summary()
        
    
    def init_root_dir(self):
        dataset = self.config['base']['dataset']
        model = self.config['base']['model']
        if self.enable_QAMA:
            root_dir = f'./results/{dataset}-{model}-QAMA-{self.config["backend_solver"]["solver"]}'
        else:
            root_dir = f'./results/{dataset}-{model}'
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
        os.makedirs(root_dir, exist_ok=True)
        shutil.copy(self.args.config_path, os.path.join(root_dir, 'config.yaml'))
        return root_dir
    
    def paser_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("config_path", type=str, help="Path to the config file")
        args = parser.parse_args()
        assert args.config_path.endswith(".yaml"), "Config file must be a .yaml file"
        return args

    def load_config(self, config_path: str):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def load_class(self, module_path: str, class_name: str):
        module = import_module(module_path)
        class_ = getattr(module, class_name)
        return class_
    
    def init_logger(self, root_dir: str):
        logger = logging.getLogger('train_logger')
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(root_dir, 'train.log'), encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def load_runner(self):
        # load runner
        runner_path = 'common.model.' + self.config['base']['model'] + '.Runner'
        Runner = self.load_class(runner_path, 'Runner')
        runner = Runner(
            model_setting=self.config['model_setting'], 
            train_setting=self.config['train_setting'], 
            QAMA=self.enable_QAMA,
            backend_solver=self.config['backend_solver'] if self.enable_QAMA else None,
        )
        return runner
    
    def model_summary(self):
        total_params = sum(p.numel() for p in self.runner.model.parameters())
        trainable_params = sum(p.numel() for p in self.runner.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params}")
        self.logger.info(f"Trainable parameters: {trainable_params}")
        
        if self.enable_QAMA:
            self.logger.info(f"Enabled QAMA with solvers: {self.config['backend_solver']['solver']}")
        else:
            self.logger.info("QAMA is not enabled.")
            
        self.logger.info(f"Model architecture: {self.runner.model}")
    
    def load_dataset(self):
        # load dataset
        dataload_path = 'common.data.' + self.config['base']['dataset'] + '.dataload'
        data_module = self.load_class(dataload_path, 'Dataset')(self.config['train_setting']['batch_size'])
        data_module.perpare_data()
        data_module.setup('fit')
        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()
        return train_dataloader, val_dataloader
    
    def dataset_summary(self):
        train_size = len(self.train_dataloader.dataset)
        val_size = len(self.val_dataloader.dataset)
        self.logger.info(f"Training dataset size: {train_size}")
        self.logger.info(f"Validation dataset size: {val_size}")
    
    def train(self):
        self.logger.info("Starting training process...")
        # train
        optimizer = torch.optim.Adam(self.runner.model.parameters(), lr=self.config['train_setting']['learning_rate'])
        record_train_epoch, record_val_epoch = {}, {}
        for epoch in tqdm(range(self.config['train_setting']['num_epochs']), desc="Training Epochs"):
            # training
            self.runner.model.train()
            epoch_train_output = {}        
            for batch in tqdm(self.train_dataloader, desc="Training Batches", leave=False, total=len(self.train_dataloader)):
                output = self.runner.training_step(batch)
                loss = output['loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                for metrics_name, value in output.items():
                    if metrics_name not in epoch_train_output:
                        epoch_train_output[metrics_name] = []
                    epoch_train_output[metrics_name].append(value.item() if isinstance(value, torch.Tensor) else value)
            
            for metrics_name, values in epoch_train_output.items():
                avg_value = sum(values) / len(values)
                if metrics_name not in record_train_epoch:
                    record_train_epoch[metrics_name] = []
                record_train_epoch[metrics_name].append(avg_value)
                self.logger.info(f"Epoch {epoch+1} - Train {metrics_name}: {avg_value:.4f}")
                
                
            # validation
            self.runner.model.eval()
            epoch_val_output = {}
            with torch.no_grad():
                for batch in tqdm(self.val_dataloader, desc="Validation Batches", leave=False, total=len(self.val_dataloader)):
                    output = self.runner.validation_step(batch)
                    
                    for metrics_name, value in output.items():
                        if metrics_name not in epoch_val_output:
                            epoch_val_output[metrics_name] = []
                        epoch_val_output[metrics_name].append(value.item() if isinstance(value, torch.Tensor) else value)
                        
            for metrics_name, values in epoch_val_output.items():
                avg_value = sum(values) / len(values)
                if metrics_name not in record_val_epoch:
                    record_val_epoch[metrics_name] = []
                record_val_epoch[metrics_name].append(avg_value)
                self.logger.info(f"Epoch {epoch+1} - Val {metrics_name}: {avg_value:.4f}")
                
            
            if not os.path.exists(os.path.join(self.root_dir, 'saved_model')):
                os.makedirs(os.path.join(self.root_dir, 'saved_model'))
            torch.save(self.runner.model.state_dict(), os.path.join(self.root_dir, 'saved_model', f'epoch_{epoch+1}.pth'))
        
        # save training records
        with open(os.path.join(self.root_dir, 'train_record.json'), 'w+') as f:
            json.dump(record_train_epoch, f, indent=4)
        with open(os.path.join(self.root_dir, 'val_record.json'), 'w+') as f:
            json.dump(record_val_epoch, f, indent=4)
        
        self.logger.info("Training completed successfully.")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
