import tempfile

import pytest
import torch
from torch import nn

from audiotools import ml
from audiotools import util


class Model(ml.BaseModel):
    def __init__(self, arg1: float = 1.0):
        super().__init__()
        self.arg1 = arg1
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def test_trainer():
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, N):
            super().__init__()
            self.N = N

        def __len__(self):
            return self.N

        def __getitem__(self, idx):
            return torch.randn(1, 1)

    class Trainer(ml.BaseTrainer):
        def train_loop(self, engine, batch):
            return {"loss": torch.randn(1)}

        def val_loop(self, engine, batch):
            return {"loss": torch.randn(1)}

        def checkpoint(self, engine):
            is_best = self.is_best(engine, "loss/val")
            top_k = self.top_k(engine, "loss/val", 5)

    train_data = torch.utils.data.DataLoader(Dataset(10), batch_size=1)
    val_data = torch.utils.data.DataLoader(Dataset(10), batch_size=1)

    with tempfile.TemporaryDirectory() as d:
        with util.chdir(d):
            writer = torch.utils.tensorboard.SummaryWriter(".")
            trainer = Trainer(writer=writer, a=1, record_memory=True)

            assert trainer.a == 1
            trainer.run(train_data, val_data, num_epochs=5)

            state_dict = trainer.state_dict()
            assert "trainer" in state_dict
            assert "validator" in state_dict

            trainer.load_state_dict(state_dict)

            with tempfile.NamedTemporaryFile(suffix=".pth") as f:
                trainer.save(f.name)

            assert trainer.state == trainer.trainer.state

            trainer = Trainer(quiet=True)
            trainer.run(train_data, val_data, num_epochs=5)

            # Test a non-local rank trainer
            trainer = Trainer(rank=1)
            trainer.run(train_data, val_data, num_epochs=5)


def test_timer():
    timer = ml.trainer.SimpleTimer()
    timer(message="Some message")
