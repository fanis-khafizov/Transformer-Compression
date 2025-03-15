import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

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