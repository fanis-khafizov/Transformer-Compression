import os
from dotenv import load_dotenv

import wandb
import torch
from torch.utils.data import DataLoader

from utils import get_datasets, get_device
from config import ExperimentConfig
from optimizers import CAdamW
from experiment import Experiment

# Load environment variables
load_dotenv()

# Authenticate with W&B using API key from environment variable
wandb.login(key=os.environ.get("WANDB_API_KEY"))

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    train_inputs, val_inputs, test_inputs = get_datasets()
    trainloader = DataLoader(train_inputs, batch_size=2, shuffle=True)
    testloader = DataLoader(val_inputs, batch_size=1, shuffle=False)
    device = get_device()

    train_config = {
        'param_usage': 0.01,
        'num_restarts': 1,
        'num_epochs': 1,
    }
    # Извлечение настроек из train_config
    param_usage = train_config['param_usage']
    num_restarts = train_config['num_restarts']
    num_epochs = train_config['num_epochs']

    # Создание списка объектов конфигов
    configs = [
        # ExperimentConfig(train_config, name='TopK_0.01', strategy='TopK', lr=0.01, optimizer=CAdamW),
        # ExperimentConfig(train_config, name='TopK_EF_0.0001', strategy='TopK', error_correction='EF', lr=0.0001, optimizer=CAdamW),
        # ExperimentConfig(train_config, name='SCAM_TopK_0.001', strategy='SCAM_TopK', lr=0.001, optimizer=CAdamW),
        ExperimentConfig(train_config, name='ImpK_b_EF_0.0001', strategy='ImpK', error_correction='EF', update_task='mirror_descent', update_kwargs={'lambda_value':1e-6,'eta':1e3,'num_steps':50}, lr=0.0001, optimizer=CAdamW),
        # ExperimentConfig(train_config, name='ImpK_c_EF_0.0001', strategy='ImpK', error_correction='EF', update_task='gradient_descent', update_kwargs={'scale':2.0,'eta':1e7,'num_steps':50}, lr=0.0001, optimizer=CAdamW),
        # ExperimentConfig(train_config, name='SCAM_b_EF_0.001', strategy='SCAM', error_correction='EF', update_task='mirror_descent', update_kwargs={'lambda_value':1e-5,'eta':1e3,'num_steps':50}, lr=0.001, optimizer=CAdamW),
        # ExperimentConfig(train_config, name='SCAM_c_EF_0.001', strategy='SCAM', error_correction='EF', update_task='gradient_descent', update_kwargs={'scale':2.0,'eta':1e7,'num_steps':50}, lr=0.001, optimizer=CAdamW),
    ]
    for cfg in configs:
        experiment = Experiment(cfg, trainloader, testloader, device, param_usage, num_epochs, num_restarts)
        experiment.run()
