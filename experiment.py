import wandb
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from logger import TrainerLogger
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
        self.logger = TrainerLogger(config.name, param_usage)

    def run(self):
        # Инициализация W&B для этого эксперимента
        self.config.init_wandb()

        for restart in range(self.num_restarts):
            set_seed(52 + restart)

            # Создание модели и компрессора
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            model_config = GPT2Config(vocab_size=tokenizer.vocab_size)
            model = GPT2LMHeadModel(model_config).to(self.device)

            compressor = compressors.Compressor(
                model=model,
                k=self.param_usage,
                strategy=self.config.strategy,
                error_correction=self.config.error_correction,
                update_task=self.config.update_task,
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
                lr=self.config.lr,
                eta=self.config.eta,
                num_steps=self.config.num_steps,
                device=self.device,
                restart=restart,
                logger=self.logger
            )

        # Завершение W&B
        wandb.finish()
        # Сохранение и визуализация результатов
        self.logger.save_csv()
        self.logger.plot(plot_and_save_results)
