import time

import torch
from torch.utils.tensorboard import SummaryWriter

from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when


def test_all_decorators():
    i = 0
    rank = 0
    max_iters = 100

    writer = SummaryWriter("/tmp/logs")
    tracker = Tracker(lambda: i, writer, log_file="/tmp/log.txt")

    train_data = range(100)
    val_data = range(100)

    @tracker.log("train", "value", history=False)
    @tracker.track("train", max_iters, i)
    @timer()
    def train_loop():
        time.sleep(0.01)
        return {
            "loss": torch.exp(torch.FloatTensor([-i / 100])),
            "mel": torch.exp(torch.FloatTensor([-i / 100])),
            "stft": torch.exp(torch.FloatTensor([-i / 100])),
            "waveform": torch.exp(torch.FloatTensor([-i / 100])),
        }

    @tracker.track("val", len(val_data))
    @timer()
    def val_loop():
        time.sleep(0.01)
        return {
            "loss": torch.exp(torch.FloatTensor([-i / 100])),
            "mel": torch.exp(torch.FloatTensor([-i / 100])),
            "stft": torch.exp(torch.FloatTensor([-i / 100])),
            "waveform": torch.exp(torch.FloatTensor([-i / 100])),
        }

    @when(lambda: i % 1000 == 0 and rank == 0)
    @torch.no_grad()
    def save_samples():
        tracker.print("Saving samples to TensorBoard.")

    @when(lambda: i % 100 == 0 and rank == 0)
    def checkpoint():
        save_samples()
        if tracker.is_best("val", "mel"):
            tracker.print("Best model so far.")
        tracker.print("Saving to /runs/exp1")

        state_dict = tracker.state_dict()
        state_dict["i"] = i
        tracker.done("val", f"Iteration {i}")

    @when(lambda: i % 100 == 0)
    @tracker.log("val", "mean")
    @torch.no_grad()
    def validate():
        for _ in range(len(val_data)):
            output = val_loop()
        return output

    with tracker.live:
        for i in range(max_iters):
            validate()
            checkpoint()
            train_loop()

    tracker.load_state_dict(tracker.state_dict())
