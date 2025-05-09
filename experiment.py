import wandb
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from train import train
from utils import set_seed, plot_and_save_results
import compressors

class Experiment:
    def __init__(self, config, trainloader, testloader, device, param_usage, num_epochs, num_restarts):
        self.config = config
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.param_usage = param_usage
        self.num_epochs = num_epochs
        self.num_restarts = num_restarts

    def run(self):
        for restart in range(self.num_restarts):
            wandb.init(
                project=self.config.project_name,
                name=f"{self.config.name}_restart_{restart}",
                config={**self.config.train_config, **self.config.to_dict()},
                reinit=True
            )

            set_seed(52 + restart)

            # Создание модели и компрессора
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            model_config = GPT2Config(vocab_size=tokenizer.vocab_size)
            model = GPT2LMHeadModel(model_config)
            model.to(self.device)
            model = torch.compile(model)

            compressor = compressors.Compressor(
                model=model,
                k=self.param_usage,
                strategy=self.config.strategy,
                error_correction=self.config.error_correction,
                update_task=self.config.update_task,
                lr=self.config.lr,
                update_kwargs=self.config.update_kwargs
            )

            # Instantiate optimizer from config
            optimizer = self.config.optimizer(
                compressor=compressor,
                lr=self.config.lr,
                **self.config.optimizer_kwargs
            )

            # Тренировка и валидация с логированием через logger
            train_losses, train_ppls, val_losses, val_ppls = train(
                model=model,
                optimizer=optimizer,
                compressor=compressor,
                trainloader=self.trainloader,
                testloader=self.testloader,
                num_epochs=self.num_epochs,
                device=self.device,
            )

        # Завершение W&B
        wandb.finish()
