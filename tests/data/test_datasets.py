import numpy as np
import torch

import audiotools
from audiotools import AudioSignal
from audiotools.data import transforms as tfm


def test_static_shared_args():
    dataset = audiotools.data.datasets.CSVDataset(
        44100,
        n_examples=100,
        csv_files=["tests/audio/spk.csv"],
    )

    for nw in (0, 1, 2):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=nw,
            collate_fn=dataset.collate,
        )

        targets = {"dur": [dataloader.dataset.duration], "sr": [44100]}
        observed = {"dur": [], "sr": []}

        sample_rates = [8000, 16000, 44100]

        for batch in dataloader:
            dur = np.random.rand()
            sr = int(np.random.choice(sample_rates))

            # Change attributes in the shared dict.
            # Later we'll make sure they actually worked.
            dataloader.dataset.duration = dur
            dataloader.dataset.sample_rate = sr

            # Record observations from the batch and the signal.
            targets["dur"].append(dur)
            observed["dur"].append(batch["signal"].signal_duration)

            targets["sr"].append(sr)
            observed["sr"].append(batch["signal"].sample_rate)

        # You aren't guaranteed that every requested attribute setting gets to every
        # worker in time, but you can expect that every output attribute
        # is in the requested attributes, and that it happens at least twice.
        for k in targets:
            _targets = targets[k]
            _observed = observed[k]

            num_succeeded = 0
            for val in np.unique(_observed):
                assert np.any(np.abs(np.array(_targets) - val) < 1e-3)
                num_succeeded += 1

            assert num_succeeded >= 2


# This transform just adds the ID of the object, so we
# can see if it's the same across processes.
class IDTransform(audiotools.data.transforms.BaseTransform):
    def __init__(self, id):
        super().__init__(["id"])
        self.id = id

    def _instantiate(self, state):
        return {"id": self.id}


def test_shared_transform():
    for nw in (0, 1, 2):
        transform = IDTransform(1)
        dataset = audiotools.data.datasets.CSVDataset(
            44100,
            n_examples=10,
            csv_files=["tests/audio/spk.csv"],
            transform=transform,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=nw,
            collate_fn=dataset.collate,
        )

        targets = {"id": [transform.id]}
        observed = {"id": []}

        for batch in dataloader:
            kwargs = batch["transform_args"]
            new_id = np.random.randint(100)

            # Create a new transform with a different ID.
            # This gets propagated to all processes.
            transform = IDTransform(new_id)
            dataloader.dataset.transform = transform

            targets["id"].append(new_id)
            observed["id"].append(kwargs["IDTransform"]["id"])

        for k in targets:
            _targets = [int(x) for x in targets[k]]
            _observed = [int(x.item()) for x in observed[k]]

            num_succeeded = 0
            for val in np.unique(_observed):
                assert any([x == val for x in _targets])
                num_succeeded += 1
            assert num_succeeded >= 2


def test_csv_dataset():
    transform = tfm.Compose(
        [
            tfm.VolumeNorm(),
            tfm.Silence(prob=0.5),
        ],
    )
    dataset = audiotools.data.datasets.CSVDataset(
        44100,
        n_examples=100,
        csv_files=["tests/audio/spk.csv"],
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

        mask = kwargs["Compose"]["0.Silence"]["mask"]

        zeros_ = torch.zeros_like(signal[mask].audio_data)
        original_ = original[~mask].audio_data

        assert torch.allclose(signal[mask].audio_data, zeros_)
        assert torch.allclose(signal[~mask].audio_data, original_)


def test_dataset_pipeline():
    transform = tfm.Compose(
        [
            tfm.RoomImpulseResponse(csv_files=["tests/audio/irs.csv"]),
            tfm.BackgroundNoise(csv_files=["tests/audio/noises.csv"]),
        ]
    )
    dataset = audiotools.data.datasets.CSVDataset(
        44100, 10, csv_files=["tests/audio/spk.csv"], transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=0, batch_size=1, collate_fn=dataset.collate
    )
    for batch in dataloader:
        batch = audiotools.core.util.prepare_batch(batch, device="cpu")
        kwargs = batch["transform_args"]
        signal = batch["signal"]
        batch = dataset.transform(signal, **kwargs)
