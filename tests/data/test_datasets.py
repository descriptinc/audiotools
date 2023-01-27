import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import audiotools
from audiotools.data import transforms as tfm


def test_audio_dataset():
    transform = tfm.Compose(
        [
            tfm.VolumeNorm(),
            tfm.Silence(prob=0.5),
        ],
    )
    loader = audiotools.data.datasets.AudioLoader(
        sources=["tests/audio/spk.csv"],
        transform=transform,
    )
    dataset = audiotools.data.datasets.AudioDataset(
        loader,
        44100,
        n_examples=100,
        transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        num_workers=0,
        collate_fn=dataset.collate,
    )
    for batch in dataloader:
        kwargs = batch["transform_args"]
        signal = batch["signal"]
        original = signal.clone()

        signal = dataset.transform(signal, **kwargs)
        original = dataset.transform(original, **kwargs)

        mask = kwargs["Compose"]["1.Silence"]["mask"]

        zeros_ = torch.zeros_like(signal[mask].audio_data)
        original_ = original[~mask].audio_data

        assert torch.allclose(signal[mask].audio_data, zeros_)
        assert torch.allclose(signal[~mask].audio_data, original_)


def test_aligned_audio_dataset():
    with tempfile.TemporaryDirectory() as d:
        dataset_dir = Path(d)
        audiotools.util.generate_chord_dataset(
            max_voices=8, num_items=3, output_dir=dataset_dir
        )
        loaders = [
            audiotools.data.datasets.AudioLoader([dataset_dir / f"track_{i}"])
            for i in range(3)
        ]
        dataset = audiotools.data.datasets.AudioDataset(
            loaders, 44100, n_examples=1000, aligned=True, shuffle_loaders=True
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            num_workers=0,
            collate_fn=dataset.collate,
        )

        # Make sure the voice tracks are aligned.
        for batch in dataloader:
            paths = []
            for i in range(len(loaders)):
                _paths = [p.split("/")[-1] for p in batch[i]["path"]]
                paths.append(_paths)
            paths = np.array(paths)
            for i in range(paths.shape[1]):
                col = paths[:, i]
                col = col[col != "none"]
                assert np.all(col == col[0])


def test_dataset_pipeline():
    transform = tfm.Compose(
        [
            tfm.RoomImpulseResponse(sources=["tests/audio/irs.csv"]),
            tfm.BackgroundNoise(sources=["tests/audio/noises.csv"]),
        ]
    )
    loader = audiotools.data.datasets.AudioLoader(sources=["tests/audio/spk.csv"])
    dataset = audiotools.data.datasets.AudioDataset(
        loader,
        44100,
        n_examples=10,
        transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=0, batch_size=1, collate_fn=dataset.collate
    )
    for batch in dataloader:
        batch = audiotools.core.util.prepare_batch(batch, device="cpu")
        kwargs = batch["transform_args"]
        signal = batch["signal"]
        batch = dataset.transform(signal, **kwargs)
