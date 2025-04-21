import torch
from tqdm import tqdm, trange

def train(model, optimizer, compressor, trainloader, testloader, num_epochs, lr, eta, num_steps, device, quiet=False):
    train_losses, train_ppls = [], []
    val_losses, val_ppls = [], []
    for epoch in trange(num_epochs):
        if not quiet:
            tqdm.write('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0.0
        train_ppl = 0.0
        
        for batch_idx, batch in enumerate(tqdm(trainloader)):
            batch = batch.to(device)

            if batch_idx == 0:
                # Обновляем компрессор с новым интерфейсом
                compressor.update(batch, lr, eta, num_steps)
            
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            
            for name, param in model.named_parameters():
                if 'ln' in name:
                    continue
                param.grad.copy_(compressor.compress(name, param))
            
            optimizer.step()

            train_loss += loss.item()
            train_ppl += torch.exp(loss).item()

        train_loss /= len(trainloader)
        train_ppl /= len(trainloader)
        train_losses.append(train_loss)
        train_ppls.append(train_ppl)
        

        # Validation
        model.eval()
        val_loss = 0.0
        val_ppl = 0.0
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
        if not quiet:
            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")
            print(f"Epoch {epoch+1}, Train PPL: {train_ppl}, Val PPL: {val_ppl}")
    
    
    return train_losses, train_ppls, val_losses, val_ppls