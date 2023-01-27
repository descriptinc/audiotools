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
    loader = audiotools.data.datasets.AudioLoader(sources=["tests/audio/spk.csv"])
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
