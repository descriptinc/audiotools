import time

import numpy as np
import torch
from rich.console import Console
from rich.progress import track
from rich.table import Table

from audiotools import AudioSignal
from audiotools.core import util
from audiotools.data import transforms as tfm
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import AudioLoader

transforms_to_demo = []
for x in dir(tfm):
    if hasattr(getattr(tfm, x), "transform"):
        if x not in ["Compose", "Choose", "Repeat", "RepeatUpTo"]:
            transforms_to_demo.append(x)


def run(batch_size=64, duration=5.0, device="cuda"):
    times = {}

    for transform_name in track(transforms_to_demo):
        kwargs = {}
        if transform_name == "BackgroundNoise":
            kwargs["sources"] = ["tests/audio/noises.csv"]
        if transform_name == "RoomImpulseResponse":
            kwargs["sources"] = ["tests/audio/irs.csv"]
        if "Quantization" in transform_name:
            kwargs["channels"] = ("choice", [8, 16, 32])

        transform_cls = getattr(tfm, transform_name)
        t = transform_cls(prob=1.0, **kwargs)

        loader = AudioLoader(sources=["tests/audio/spk.csv"])
        dataset = AudioDataset(
            loader,
            44100,
            batch_size * 10,
            duration,
            transform=t,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=dataset.collate
        )
        batch = next(iter(dataloader))
        batch = util.prepare_batch(batch, device)

        with torch.no_grad():
            start_time = time.time()
            output = t(batch["signal"], **batch["transform_args"])
            torch.cuda.synchronize()
            elapsed = time.time() - start_time

        times[transform_name] = elapsed

    table = Table(expand=False)

    for k, v in times.items():
        row_args = [
            k,
        ]
        row_args.append(str(v))
        table.add_row(*row_args)

    console = Console()
    console.print(table)


if __name__ == "__main__":
    run(64, 5.0, "cuda")
