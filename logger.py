import os
import csv
import datetime
import wandb

class TrainerLogger:
    def __init__(self, name: str, param_usage: float):
        self.name = name
        self.param_usage = param_usage
        self.records = []  # список словарей: {'epoch', 'restart', 'train_loss', 'train_ppl', 'val_loss', 'val_ppl'}

    def log(self, epoch: int, restart: int, train_loss: float, train_ppl: float, val_loss: float, val_ppl: float):
        # Логирование в W&B
        wandb.log({
            'train/loss': train_loss,
            'train/ppl': train_ppl,
            'val/loss': val_loss,
            'val/ppl': val_ppl
        }, step=epoch)
        # Сохранение записи локально
        self.records.append({
            'epoch': epoch,
            'restart': restart,
            'train_loss': train_loss,
            'train_ppl': train_ppl,
            'val_loss': val_loss,
            'val_ppl': val_ppl
        })

    def save_csv(self, log_dir: str = 'logs'):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{self.name}_{self.param_usage*100:.0f}%_{date}.csv"
        path = os.path.join(log_dir, fname)
        fieldnames = ['epoch', 'restart', 'train_loss', 'train_ppl', 'val_loss', 'val_ppl']
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.records:
                writer.writerow(rec)

    def plot(self, plot_func):
        # Группируем по рестартам и эпохам
        train_losses = []
        train_ppls = []
        val_losses = []
        val_ppls = []
        # Преобразуем записи в структуру: списки по рестартам
        restarts = max(rec['restart'] for rec in self.records) + 1 if self.records else 0
        epochs = max(rec['epoch'] for rec in self.records) + 1 if self.records else 0
        # Инициализация
        train_losses = [[0.0]*epochs for _ in range(restarts)]
        train_ppls = [[0.0]*epochs for _ in range(restarts)]
        val_losses = [[0.0]*epochs for _ in range(restarts)]
        val_ppls = [[0.0]*epochs for _ in range(restarts)]
        for rec in self.records:
            r = rec['restart']
            e = rec['epoch']
            train_losses[r][e] = rec['train_loss']
            train_ppls[r][e] = rec['train_ppl']
            val_losses[r][e] = rec['val_loss']
            val_ppls[r][e] = rec['val_ppl']
        # Вызов функции построения графиков
        plot_func(train_losses, train_ppls, val_losses, val_ppls, self.name)
