import train
import compressors
from utils import set_seed, get_datasets, get_device, plot_and_save_results
import torch.optim as optim
import torch.nn as nn
import numpy as np
import datetime
import os
import csv
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
import wandb
from dotenv import load_dotenv
load_dotenv()
# Авторизация в W&B через API ключ из переменной окружения
wandb.login(key=os.environ.get("WANDB_API_KEY"))
from compress_config import CompressionConfig
from experiment import Experiment, ExperimentManager

if __name__ == "__main__":
    train_inputs, val_inputs, test_inputs = get_datasets()
    trainloader = DataLoader(train_inputs, batch_size=1, shuffle=True)
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
        CompressionConfig(train_config, name='TopK', strategy='TopK', lr=0.01),
        CompressionConfig(train_config, name='TopK_EF', strategy='TopK', error_correction='EF', lr=0.01),
        CompressionConfig(train_config, name='ImpK_b_EF', strategy='ImpK', error_correction='EF', update_task='mirror_descent_full', update_kwargs={'lambda_value':1e-6,'start':'ones'}, lr=0.01, eta=1e3, num_steps=50),
        CompressionConfig(train_config, name='ImpK_c_EF', strategy='ImpK', error_correction='EF', update_task='gradient_descent_full', update_kwargs={'scale':2.0,'start':'ones'}, lr=0.01, eta=1e7, num_steps=50),
        CompressionConfig(train_config, name='SCAM_b_EF', strategy='SCAM', error_correction='EF', update_task='mirror_descent_full', update_kwargs={'lambda_value':1e-6,'start':'ones'}, lr=0.01, eta=1e3, num_steps=50),
        CompressionConfig(train_config, name='SCAM_c_EF', strategy='SCAM', error_correction='EF', update_task='gradient_descent_full', update_kwargs={'scale':2.0,'start':'ones'}, lr=0.01, eta=1e7, num_steps=50),
    ]
    experiments = [
        Experiment(cfg, trainloader, testloader, device, param_usage, num_epochs, num_restarts)
        for cfg in configs
    ]
    manager = ExperimentManager(experiments)
    manager.run_all()