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
    trainloader = DataLoader(train_inputs, batch_size=1, shuffle=True)
    testloader = DataLoader(val_inputs, batch_size=1, shuffle=False)
    device = get_device()

    config = {
        'param_usage': 0.01,
        'num_restarts': 1,
        'num_epochs': 25,
    }

    compress_configs = [
        # {
        #     'compression_type': 'TopK',
        #     'lr': 0.005,
        # },
        # {
        #     'compression_type': 'TopK',
        #     'lr': 0.01,
        # },
        # {
        #     'compression_type': 'TopK',
        #     'lr': 0.02,
        # },
        # {
        #     'compression_type': 'TopK_EF',
        #     'lr': 0.005,
        # },
        # {
        #     'compression_type': 'TopK_EF',
        #     'lr': 0.01,
        # },
        # {
        #     'compression_type': 'TopK_EF',
        #     'lr': 0.02,
        # },
        # {
        #     'compression_type': 'ImpK_b_EF',
        #     'start': 'ones',
        #     'lr': 0.01,
        #     'eta': 1000000.,
        #     'num_steps': 25,
        # },
        # {
        #     'compression_type': 'SCAM_b_EF',
        #     'start': 'ones',
        #     'lr': 0.01,
        #     'eta': 1000000.,
        #     'num_steps': 25,
        # },
        # {
        #     'compression_type': 'ImpK_b',
        #     'start': 'ones',
        #     'lr': 0.01,
        #     'eta': 7.,
        #     'num_steps': 25,
        # },
        # {
        #     'compression_type': 'ImpK_b',
        #     'start': 'ones',
        #     'lr': 0.02,
        #     'eta': 2.,
        #     'num_steps': 20,
        # },
        # {
        #     'compression_type': 'ImpK_b',
        #     'start': 'abs',
        #     'lr': 0.01,
        #     'eta': 2.,
        #     'num_steps': 20,
        # },
        # {
        #     'compression_type': 'ImpK_c_EF21',
        #     'start': 'ones',
        #     'lr': 0.01,
        #     'eta': 1000000.,
        #     'num_steps': 25,
        #     'scale': 1.0,
        # },
        # {
        #     'compression_type': 'ImpK_c',
        #     'start': 'ones',
        #     'lr': 0.015,
        #     'eta': 1000000.,
        #     'num_steps': 20,
        #     'scale': 1.0,
        # },
        # {
        #     'compression_type': 'ImpK_c',
        #     'start': 'ones',
        #     'lr': 0.02,
        #     'eta': 1000000.,
        #     'num_steps': 20,
        #     'scale': 1.0,
        # },
        # {
        #     'compression_type': 'ImpK_c_EF',
        #     'start': 'ones',
        #     'lr': 0.001,
        #     'eta': 1000000.,
        #     'num_steps': 25,
        #     'scale': 1.0,
        # },
        # {
        #     'compression_type': 'SCAM_c_EF',
        #     'start': 'ones',
        #     'lr': 0.01,
        #     'eta': 1000000.,
        #     'num_steps': 25,
        #     'scale': 1.0,
        # },
    ]


    train_log, train_ppl_log = {}, {}
    test_log, test_ppl_log = {}, {}

    param_usage = config['param_usage']
    num_restarts = config['num_restarts']
    num_epochs = config['num_epochs']

    for compress_config in compress_configs:
        compression_type = compress_config['compression_type']

        start = compress_config.get('start', '')
        lr = compress_config.get('lr', '')
        eta = compress_config.get('eta', '')
        num_steps = compress_config.get('num_steps', '')
        scale=compress_config.get('scale', '')

        name = f'{compression_type}_{start}_{lr}'

        train_log[name], train_ppl_log[name], test_log[name], test_ppl_log[name] = [], [], [], []
        
        for num_restart in range(num_restarts):
            set_seed(52 + num_restart)
            
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            config = GPT2Config(vocab_size=tokenizer.vocab_size)
            model = GPT2LMHeadModel(config)
            model.loss_type = "lm"
            model = model.to(device)

            if compression_type == 'TopK':
                compressor = compressors.TopK(param_usage)
            elif compression_type == 'TopK_EF':
                compressor = compressors.TopK_EF(param_usage, model)
            elif compression_type == 'TopK_EF21':
                compressor = compressors.TopK_EF21(param_usage, model)
            elif compression_type == 'RandK':
                compressor = compressors.RandK(param_usage)
            elif compression_type == 'ImpK_b':
                compressor = compressors.ImpK_b(model, param_usage, start=start)
            elif compression_type == 'ImpK_b_EF':
                compressor = compressors.ImpK_b_EF(model, param_usage, start=start)
            elif compression_type == 'ImpK_b_EF21':
                compressor = compressors.ImpK_b_EF21(model, param_usage, start=start)
            elif compression_type == 'ImpK_c':
                compressor = compressors.ImpK_c(model, param_usage, start=start, scale=scale)
            elif compression_type == 'ImpK_c_EF':
                compressor = compressors.ImpK_c_EF(model, param_usage, start=start, scale=scale)
            elif compression_type == 'ImpK_c_EF21':
                compressor = compressors.ImpK_c_EF21(model, param_usage, start=start, scale=scale)
            elif compression_type == 'SCAM_b_EF':
                compressor = compressors.SCAM_b_EF(model, param_usage, start=start)
            elif compression_type == 'SCAM_c_EF':
                compressor = compressors.SCAM_c_EF(model, param_usage, start=start, scale=scale)
            else:
                raise ValueError(f"Unknown compression type: {compression_type}")
            
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
            print(f"# Compression type: {compression_type}, start: {start}, num_restart: {num_restart}, lr: {lr}, eta: {eta}, num_steps: {num_steps}")
            print("# Train Loss")
            print(train_losses)
            print("# Train Perplexity")
            print(train_ppls)
            print("# Test Loss")
            print(val_losses)
            print("# Test Perplexity")
            print(val_ppls)
            train_log[name].append(train_losses)
            train_ppl_log[name].append(train_ppls)
            test_log[name].append(val_losses)
            test_ppl_log[name].append(val_ppls)

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
        fieldnames = ['type', 'train_log', 'train_acc', 'test_log', 'test_acc', 'epoch']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for compress_config in compress_configs:
            compression_type = compress_config['compression_type']
            start = compress_config.get('start', '')
            lr = compress_config.get('lr', '')
            eta = compress_config.get('eta', '')
            num_steps = compress_config.get('num_steps', '')

            name = f'{compression_type}_{param_usage*100:.0f}%_{lr}_{"EF" if "EF" in compression_type else ""}'

            for epoch in range(num_epochs):
                for restart in range(num_restarts):
                    dict_name = f'{compression_type}_{start}_{lr}'
                    writer.writerow({
                        'type': name,
                        'epoch': epoch,
                        'train_log': train_log[dict_name][restart][epoch],
                        'train_acc': train_ppl_log[dict_name][restart][epoch],
                        'test_log': test_log[dict_name][restart][epoch],
                        'test_acc': test_ppl_log[dict_name][restart][epoch]
                    })

    fig_train, axs_train = plt.subplots(1, 2, figsize=(16, 7))
    fig_test, axs_test = plt.subplots(1, 2, figsize=(16, 7))


    for compress_config in compress_configs:
        compression_type = compress_config['compression_type']

        start = compress_config.get('start', '')
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