import wandb
from optimizers import CSGD


# -----------------------------
class ExperimentConfig:
    def __init__(
        self,
        train_config: dict,
        name: str = None,
        strategy: str = None,
        error_correction: str = None,
        update_task: str = None,
        update_kwargs: dict = None,
        lr: float = None,
        eta: float = None,
        num_steps: int = None,
        project_name: str = "Transformer-Compression",
        optimizer = CSGD,
        optimizer_kwargs: dict = None
    ):
        self.train_config = train_config
        self.name = name
        self.strategy = strategy
        self.error_correction = error_correction
        self.update_task = update_task
        self.update_kwargs = update_kwargs or {}
        self.lr = lr
        self.eta = eta
        self.num_steps = num_steps
        self.project_name = project_name
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}

    def to_dict(self):
        return {
            "name": self.name,
            "strategy": self.strategy,
            "error_correction": self.error_correction,
            "update_task": self.update_task,
            "update_kwargs": self.update_kwargs,
            "lr": self.lr,
            "eta": self.eta,
            "num_steps": self.num_steps,
            "optimizer": self.optimizer.__name__,
            "optimizer_kwargs": self.optimizer_kwargs,
        }