import time

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from audiotools.core import util
from audiotools.data import transforms as tfm
from audiotools.data.datasets import CSVDataset


def run(batch_size=64, duration=5.0, device="cuda"):
    transform = tfm.Compose(
        [
            tfm.RoomImpulseResponse(csv_files=["tests/audio/irs.csv"]),
            tfm.BackgroundNoise(csv_files=["tests/audio/noises.csv"]),
        ]
    )
    dataset = CSVDataset(
        44100, 1000, duration, csv_files=["tests/audio/spk.csv"], transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=16, batch_size=batch_size, collate_fn=dataset.collate
    )

    timings = []
    end_to_end_time = time.time()

    for batch in dataloader:
        batch = util.prepare_batch(batch, device=device)
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
