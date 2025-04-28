import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
import os
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_datasets():
    # Загрузка датасета
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
    train_texts = dataset["train"]["text"]
    val_texts = dataset["validation"]["text"]
    test_texts = dataset["test"]["text"]

    # Инициализация токенайзера и модели с нуля
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_texts(texts):
        texts = [text.strip() for text in texts if len(text.strip()) > 0]
        encodings = tokenizer('\n'.join(texts), return_tensors="pt")
        input_ids = encodings.input_ids

        max_length = 1024
        input_chunks = [input_ids[:, i:i + max_length] for i in range(0, input_ids.size(1), max_length)]
        inputs = torch.cat(input_chunks[:-1], dim=0)
        
        return inputs

    train_inputs = tokenize_texts(train_texts)
    val_inputs = tokenize_texts(val_texts)
    test_inputs = tokenize_texts(test_texts)
    return train_inputs, val_inputs, test_inputs

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def plot_and_save_results(train_log, train_ppl_log, test_log, test_ppl_log, compress_configs, param_usage, date, figures_dir='figures'):
    """
    Plot train/test loss and perplexity for different compression configurations and save figures.
    """
    # Create figures and axes
    fig_train, axs_train = plt.subplots(1, 2, figsize=(16, 7))
    fig_test, axs_test = plt.subplots(1, 2, figsize=(16, 7))

    # Ensure figures directory exists
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Iterate over configurations and plot
    for cfg in compress_configs:
        compression_type = cfg['strategy']
        start = cfg.get('update_kwargs', {}).get('start', '')
        lr = cfg.get('lr', '')

        name = f"{compression_type}_{start}_{lr}"

        # Convert logs to numpy arrays
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

        # Plot train
        axs_train[0].plot(iters, train_loss_mean, label=f'{compression_type}, lr={lr}, start={start}')
        axs_train[0].fill_between(iters, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.1)
        axs_train[1].plot(iters, train_perplexity_mean, label=f'{compression_type}, lr={lr}, start={start}')
        axs_train[1].fill_between(iters, train_perplexity_mean - train_perplexity_std, train_perplexity_mean + train_perplexity_std, alpha=0.1)

        # Plot test
        axs_test[0].plot(iters, test_loss_mean, label=f'{compression_type}, lr={lr}, start={start}')
        axs_test[0].fill_between(iters, test_loss_mean - test_loss_std, test_loss_mean + test_loss_std, alpha=0.1)
        axs_test[1].plot(iters, test_perplexity_mean, label=f'{compression_type}, lr={lr}, start={start}')
        axs_test[1].fill_between(iters, test_perplexity_mean - test_perplexity_std, test_perplexity_mean + test_perplexity_std, alpha=0.1)

    # Customize plots
    axs_train[0].set_title(f"Train Loss Comparison, param_usage={param_usage}")
    axs_train[0].set_xlabel("Epoch")
    axs_train[0].set_ylabel("Loss")
    axs_train[0].legend()
    axs_train[0].grid()

    axs_train[1].set_title(f"Train Perplexity Comparison, param_usage={param_usage}")
    axs_train[1].set_xlabel("Epoch")
    axs_train[1].set_ylabel("Perplexity")
    axs_train[1].legend()
    axs_train[1].grid()

    axs_test[0].set_title(f"Test Loss Comparison, param_usage={param_usage}")
    axs_test[0].set_xlabel("Epoch")
    axs_test[0].set_ylabel("Loss")
    axs_test[0].legend()
    axs_test[0].grid()

    axs_test[1].set_title(f"Test Perplexity Comparison, param_usage={param_usage}")
    axs_test[1].set_xlabel("Epoch")
    axs_test[1].set_ylabel("Perplexity")
    axs_test[1].legend()
    axs_test[1].grid()

    # Save figures
    fig_train.savefig(os.path.join(figures_dir, f"train_comparison_param_usage_{param_usage}_{date}.png"))
    fig_test.savefig(os.path.join(figures_dir, f"test_comparison_param_usage_{param_usage}_{date}.png"))

    # Show plots
    fig_train.show()
    fig_test.show()