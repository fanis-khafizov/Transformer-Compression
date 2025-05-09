import torch
from tqdm import tqdm, trange
import time

import wandb

def train(model, optimizer, compressor, trainloader, testloader, num_epochs, device):
    train_losses, train_ppls = [], []
    val_losses, val_ppls = [], []
    for epoch in trange(num_epochs):
        model.train()
        train_loss = 0.0
        train_ppl = 0.0
        
        start_time = time.time()
        for batch_idx, batch in enumerate(tqdm(trainloader)):
            batch = batch.to(device)

            if batch_idx == 0:
                # Обновляем компрессор с новым интерфейсом
                compressor.update(batch)
                update_time = time.time() - start_time
                wandb.log({"train/update_time": update_time}, step=epoch)
            
            optimizer.zero_grad()

            outputs = model(batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()

            train_loss += loss.item()
            train_ppl += torch.exp(loss).item()

        train_loss /= len(trainloader)
        train_ppl /= len(trainloader)
        train_losses.append(train_loss)
        train_ppls.append(train_ppl)

        train_epoch_time = time.time() - start_time
        wandb.log({"train/epoch_time": train_epoch_time}, step=epoch)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_ppl = 0.0
        val_start_time = time.time()
        for batch in tqdm(testloader):
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            val_loss += loss.item()
            val_ppl += torch.exp(loss).item()
        
        val_loss /= len(testloader)
        val_ppl /= len(testloader)
        val_losses.append(val_loss)
        val_ppls.append(val_ppl)

        val_epoch_time = time.time() - val_start_time
        wandb.log({"val/epoch_time": val_epoch_time}, step=epoch)

        wandb.log({
            'train/loss': train_loss,
            'train/ppl': train_ppl,
            'val/loss': val_loss,
            'val/ppl': val_ppl
        }, step=epoch)
    
    
    return train_losses, train_ppls, val_losses, val_ppls