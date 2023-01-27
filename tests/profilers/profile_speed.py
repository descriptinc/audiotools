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


def run(batch_size=64, duration=5.0, device="cuda"):
    transform = tfm.Compose(
        [
            tfm.RoomImpulseResponse(csv_files=["tests/audio/irs.csv"]),
            tfm.BackgroundNoise(csv_files=["tests/audio/noises.csv"]),
        ]
    )
    loader = AudioLoader(sources=["tests/audio/spk.csv"])
    dataset = AudioDataset(
        loader,
        44100,
        n_examples=1000,
        duration=duration,
        transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=16, batch_size=batch_size, collate_fn=dataset.collate
    )

    timings = []
    end_to_end_time = None

    for batch in track(dataloader, "Generating data"):
        batch = util.prepare_batch(batch, device=device)
        # This skips the load time of the first batch.
        if end_to_end_time is None:
            end_to_end_time = time.time()
        with torch.no_grad():
            start_time = time.time()
            batch = dataset.transform(batch)
            elapsed = time.time() - start_time
            timings.append(elapsed)

    total_time = time.time() - end_to_end_time
    transform_time = np.mean(timings)

    stats = {
        "n_batches": len(dataloader),
        "batch_size": batch_size,
        "duration": duration,
        "device": device,
        "transform_time": transform_time,
        "total_time": total_time,
        "items_per_sec": (len(dataset) - batch_size) / total_time,
    }

    table = Table(expand=False)

    for k, v in stats.items():
        row_args = [
            k,
        ]
        row_args.append(str(v))
        table.add_row(*row_args)

    console = Console()
    console.print(table)


if __name__ == "__main__":
    run(64, 0.5, "cpu")
    run(64, 0.5, "cuda")
    run(64, 2.0, "cuda")
    run(64, 5.0, "cuda")
