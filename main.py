import train
import compressors
from utils import set_seed, get_datasets, get_device
import torch.optim as optim
import torch.nn as nn
import numpy as np
import datetime
import os
import csv
import matplotlib.pyplot as plt
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
# from tqdm import tqdm, trange

if __name__ == "__main__":
    train_inputs, val_inputs, test_inputs = get_datasets()
    trainloader = DataLoader(train_inputs, batch_size=16, shuffle=True)
    testloader = DataLoader(val_inputs, batch_size=1, shuffle=False)
    device = get_device()

    config = {
        'param_usage': 0.01,
        'num_restarts': 1,
        'num_epochs': 2,
    }

    compress_configs = [
        {
            'name': 'TopK',
            'strategy': 'TopK',
            'error_correction': 'none',
            'update_task': None,
            'update_kwargs': {},
            'lr': 0.01,
            'eta': 0.0,
            'num_steps': 0,
        },
        {
            'name': 'TopK_EF',
            'strategy': 'TopK',
            'error_correction': 'EF',
            'update_task': None,
            'update_kwargs': {},
            'lr': 0.01,
            'eta': 0.0,
            'num_steps': 0,
        },
        {
            'name': 'ImpK_b_EF',
            'strategy': 'ImpK',
            'error_correction': 'EF',
            'update_task': 'mirror_descent_full',
            'update_kwargs': {'lambda_value': 1e-6, 'start': 'ones'},
            'lr': 0.01,
            'eta': 1e3,
            'num_steps': 50,
        },
        {
            'name': 'ImpK_c_EF',
            'strategy': 'ImpK',
            'error_correction': 'EF',
            'update_task': 'gradient_descent_full',
            'update_kwargs': {'scale': 2.0, 'start': 'ones'},
            'lr': 0.01,
            'eta': 1e7,
            'num_steps': 50,
        },
        {
            'name': 'SCAM_b_EF',
            'strategy': 'SCAM',
            'error_correction': 'EF',
            'update_task': 'mirror_descent_full',
            'update_kwargs': {'lambda_value': 1e-6, 'start': 'ones'},
            'lr': 0.01,
            'eta': 1e3,
            'num_steps': 50,
        },
        {
            'name': 'SCAM_c_EF',
            'strategy': 'SCAM',
            'error_correction': 'EF',
            'update_task': 'gradient_descent_full',
            'update_kwargs': {'scale': 2.0, 'start': 'ones'},
            'lr': 0.01,
            'eta': 1e7,
            'num_steps': 50,
        },
    ]

    train_log, train_ppl_log = {}, {}
    test_log, test_ppl_log = {}, {}

    param_usage = config['param_usage']
    num_restarts = config['num_restarts']
    num_epochs = config['num_epochs']

    for cfg in compress_configs:
        # Читаем параметры из унифицированной конфигурации
        name = cfg['name']
        strategy = cfg['strategy']
        error_correction = cfg['error_correction']
        update_task = cfg['update_task']
        update_kwargs = cfg.get('update_kwargs', {})
        start = update_kwargs.get('start', '')
        lr = cfg['lr']
        eta = cfg.get('eta', 0.0)
        num_steps = cfg.get('num_steps', 0)
        dict_name = f'{strategy}_{start}_{lr}'

        train_log[dict_name], train_ppl_log[dict_name], test_log[dict_name], test_ppl_log[dict_name] = [], [], [], []
        
        for num_restart in range(num_restarts):
            set_seed(52 + num_restart)
            
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            config = GPT2Config(vocab_size=tokenizer.vocab_size)
            model = GPT2LMHeadModel(config)
            model.loss_type = None
            model = model.to(device)

            # Создаем компрессор по единому формату
            compressor = compressors.Compressor(
                model=model,
                k=param_usage,
                strategy=strategy,
                error_correction=error_correction,
                update_task=update_task,
                update_kwargs=update_kwargs
            )

            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

            train_losses, train_ppls, val_losses, val_ppls = train.train(
                model=model,
                optimizer=optimizer,
                compressor=compressor,
                trainloader=trainloader,
                testloader=testloader,
                num_epochs=num_epochs,
                lr=lr,
                eta=eta,
                num_steps=num_steps,
                device=device
            )
            print(f"# Compression type: {strategy}, start: {update_task}, num_restart: {num_restart}, lr: {lr}, eta: {eta}, num_steps: {num_steps}")
            print("# Train Loss")
            print(train_losses)
            print("# Train Perplexity")
            print(train_ppls)
            print("# Test Loss")
            print(val_losses)
            print("# Test Perplexity")
            print(val_ppls)
            train_log[dict_name].append(train_losses)
            train_ppl_log[dict_name].append(train_ppls)
            test_log[dict_name].append(val_losses)
            test_ppl_log[dict_name].append(val_ppls)

    print("# Train Loss")
    print(train_log)
    print("# Train Perplexity")
    print(train_ppl_log)
    print("# Test Loss")
    print(test_log)
    print("# Test Perplexity")
    print(test_ppl_log)

    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{date}.csv")

    with open(log_file, 'w', newline='') as csvfile:
        fieldnames = ['type', 'train_log', 'train_ppl', 'test_log', 'test_ppl', 'epoch']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for compress_config in compress_configs:
            strategy = compress_config['strategy']
            start = update_kwargs.get('start', '')
            lr = compress_config.get('lr', '')
            eta = compress_config.get('eta', '')
            num_steps = compress_config.get('num_steps', '')

            name = f'{strategy}_{param_usage*100:.0f}%_{lr}_{"EF" if "EF" in strategy else ""}'

            for epoch in range(num_epochs):
                for restart in range(num_restarts):
                    dict_name = f'{strategy}_{start}_{lr}'
                    writer.writerow({
                        'type': name,
                        'epoch': epoch,
                        'train_log': train_log[dict_name][restart][epoch],
                        'train_ppl': train_ppl_log[dict_name][restart][epoch],
                        'test_log': test_log[dict_name][restart][epoch],
                        'test_ppl': test_ppl_log[dict_name][restart][epoch]
                    })

    fig_train, axs_train = plt.subplots(1, 2, figsize=(16, 7))
    fig_test, axs_test = plt.subplots(1, 2, figsize=(16, 7))


    for compress_config in compress_configs:
        compression_type = compress_config['strategy']

        start = update_kwargs.get('start', '')
        lr = compress_config.get('lr', '')
        eta = compress_config.get('eta', '')
        num_steps = compress_config.get('num_steps', '')

        name = f'{compression_type}_{start}_{lr}'

        train_loss = np.array(train_log[name])
        train_loss_mean = np.mean(train_loss, axis=0)
        train_loss_std = np.std(train_loss, axis=0)
        
        train_perplexity = np.array(train_ppl_log[name])
        train_perplexity_mean = np.mean(train_perplexity, axis=0)
        train_perplexity_std = np.std(train_perplexity, axis=0)
        
        test_loss = np.array(test_log[name])
        test_loss_mean = np.mean(test_loss, axis=0)
        test_loss_std = np.std(test_loss, axis=0)
        
        test_perplexity = np.array(test_ppl_log[name])
        test_perplexity_mean = np.mean(test_perplexity, axis=0)
        test_perplexity_std = np.std(test_perplexity, axis=0)
        
        iters = list(range(len(train_loss_mean)))
        
        axs_train[0].plot(iters, train_loss_mean, label=f'{compression_type}, lr={lr}, start={start}')
        axs_train[0].fill_between(iters, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.1)
        
        axs_train[1].plot(iters, train_perplexity_mean, label=f'{compression_type}, lr={lr}, start={start}')
        axs_train[1].fill_between(iters, train_perplexity_mean - train_perplexity_std, train_perplexity_mean + train_perplexity_std, alpha=0.1)

        axs_test[0].plot(iters, test_loss_mean, label=f'{compression_type}, lr={lr}, start={start}')
        axs_test[0].fill_between(iters, test_loss_mean - test_loss_std, test_loss_mean + test_loss_std, alpha=0.1)
        
        axs_test[1].plot(iters, test_perplexity_mean, label=f'{compression_type}, lr={lr}, start={start}')
        axs_test[1].fill_between(iters, test_perplexity_mean - test_perplexity_std, test_perplexity_mean + test_perplexity_std, alpha=0.1)

    axs_train[0].set_title(f"Comparison on Train, different compression types, param_usage={param_usage}")
    axs_train[0].set_xlabel("Epoch")
    axs_train[0].set_ylabel("Loss")
    axs_train[0].legend()
    axs_train[0].grid()

    axs_train[1].set_title(f"Comparison on Train, different compression types, param_usage={param_usage}")
    axs_train[1].set_xlabel("Epoch")
    axs_train[1].set_ylabel("Perplexity")
    axs_train[1].legend()
    axs_train[1].grid()
        

    axs_test[0].set_title(f"Comparison on Test, different compression types, param_usage={param_usage}")
    axs_test[0].set_xlabel("Epoch")
    axs_test[0].set_ylabel("Loss")
    axs_test[0].legend()
    axs_test[0].grid()

    axs_test[1].set_title(f"Comparison on Test, different compression types, param_usage={param_usage}")
    axs_test[1].set_xlabel("Epoch")
    axs_test[1].set_ylabel("Perplexity")
    axs_test[1].legend()
    axs_test[1].grid()

    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check if the directory 'figures' exists, if not, create it
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Save the train plot in the 'figures' directory
    fig_train.savefig(os.path.join(figures_dir, f"train_comparison_param_usage_{param_usage}_{date}.png"))

    # Save the test plot in the 'figures' directory
    fig_test.savefig(os.path.join(figures_dir, f"test_comparison_param_usage_{param_usage}_{date}.png"))

    fig_train.show()
    fig_test.show()