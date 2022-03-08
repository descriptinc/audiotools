from pathlib import Path

import pytest
import torch

import audiotools
from audiotools import AudioSignal
from audiotools import util
from audiotools.data import transforms as tfm
from audiotools.data.datasets import CSVDataset

transforms_to_test = []
for x in dir(tfm):
    if hasattr(getattr(tfm, x), "transform"):
        if x not in ["Compose"]:
            transforms_to_test.append(x)


def _compare_transform(transform_name, signal):
    regression_data = Path(f"tests/regression/transforms/{transform_name}.wav")
    regression_data.parent.mkdir(exist_ok=True, parents=True)

    if regression_data.exists():
        regression_signal = AudioSignal(regression_data)
        regression_signal.loudness()
        signal.loudness()
        assert signal == regression_signal
    else:
        signal.write(regression_data)


@pytest.mark.parametrize("transform_name", transforms_to_test)
def test_transform(transform_name):
    seed = 0
    transform_cls = getattr(tfm, transform_name)

    kwargs = {}
    if transform_name == "BackgroundNoise":
        kwargs["csv_files"] = ["tests/audio/noises.csv"]
    if transform_name == "RoomImpulseResponse":
        kwargs["csv_files"] = ["tests/audio/irs.csv"]

    audio_path = "tests/audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)
    signal.metadata["file_loudness"] = AudioSignal(audio_path).ffmpeg_loudness().item()
    transform = transform_cls(prob=1.0, **kwargs)

    batch = transform.instantiate(seed, signal)

    batch["signal"] = signal
    batch = transform(batch)

    output = batch["signal"]
    assert isinstance(batch["signal"], AudioSignal)

    _compare_transform(transform_name, output)


def test_compose():
    seed = 0

    audio_path = "tests/audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)
    transform = tfm.Compose(
        [
            tfm.RoomImpulseResponse(csv_files=["tests/audio/irs.csv"]),
            tfm.BackgroundNoise(csv_files=["tests/audio/noises.csv"]),
        ],
    )

    batch = transform.instantiate(seed, signal)

    batch["signal"] = signal.clone()
    batch = transform(batch)
    output = batch["signal"]

    _compare_transform("Compose", output)


class DummyData(torch.utils.data.Dataset):
    def __init__(self, audio_path):
        super().__init__()

        self.audio_path = audio_path
        self.length = 100
        self.transform = tfm.Silence(prob=0.5)

    def __getitem__(self, idx):
        state = util.random_state(idx)
        signal = AudioSignal.salient_excerpt(
            self.audio_path, state=state, duration=1.0
        ).resample(44100)

        item = self.transform.instantiate(state, signal=signal)
        item["signal"] = signal

        return item

    def __len__(self):
        return self.length


def test_masking():
    dataset = DummyData("tests/audio/spk/f10_script4_produced.wav")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        num_workers=0,
        collate_fn=audiotools.data.datasets.BaseDataset.collate,
    )
    for batch in dataloader:
        batch = dataset.transform(batch)
        mask = batch["Silence"]["mask"]

        zeros = torch.zeros_like(batch["signal"][mask].audio_data)
        original = batch["original"][~mask].audio_data

        assert torch.allclose(batch["signal"][mask].audio_data, zeros)
        assert torch.allclose(batch["signal"][~mask].audio_data, original)


def test_nested_masking():
    transform = tfm.Compose(
        [
            tfm.VolumeNorm(prob=0.5),
            tfm.Silence(prob=0.9),
        ],
        prob=0.9,
    )

    dataset = CSVDataset(
        44100, 1000, 0.5, csv_files=["tests/audio/spk.csv"], transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=0, batch_size=10, collate_fn=dataset.collate
    )

    for batch in dataloader:
        batch = util.prepare_batch(batch, device="cpu")
        with torch.no_grad():
            batch = dataset.transform(batch)


if __name__ == "__main__":
    for t in transforms_to_test:
        test_transform(t)
    test_compose()
    test_masking()
    test_nested_masking()
