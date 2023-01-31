import time

import numpy as np
import torch
from flatten_dict import flatten
from flatten_dict import unflatten
from rich.console import Console
from rich.progress import track
from rich.table import Table

from audiotools import AudioSignal
from audiotools.core import util
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import AudioLoader


def collate(list_of_dicts):
    # Flatten the dictionaries to avoid recursion.
    list_of_dicts = [flatten(d) for d in list_of_dicts]
    dict_of_lists = {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}

    batch = {}
    for k, v in dict_of_lists.items():
        if isinstance(v, list):
            if all(isinstance(s, AudioSignal) for s in v):
                batch[k] = AudioSignal.batch(v, pad_signals=True)
            else:
                # Borrow the default collate fn from torch.
                batch[k] = torch.utils.data._utils.collate.default_collate(v)
    return unflatten(batch)


def run(batch_size=64, duration=5.0, device="cuda"):
    loader = AudioLoader(sources=["tests/audio/spk.csv"])
    dataset = AudioDataset(
        loader,
        44100,
        10 * batch_size,
        duration,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=16, batch_size=batch_size, collate_fn=collate
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
            batch["signal"].loudness()
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            timings.append(elapsed)

    total_time = time.time() - end_to_end_time
    loudness_time = np.mean(timings)

    stats = {
        "n_batches": len(dataloader),
        "batch_size": batch_size,
        "duration": duration,
        "device": device,
        "loudness_time": loudness_time,
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
    run(64, 5.0, "cpu")
    run(64, 5.0, "cuda")
