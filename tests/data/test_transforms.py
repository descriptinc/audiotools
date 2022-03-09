from pathlib import Path

import numpy as np
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
        if x not in ["Compose", "Choose"]:
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

    kwargs = transform.instantiate(seed, signal)
    output = transform(signal, **kwargs)
    assert isinstance(output, AudioSignal)

    _compare_transform(transform_name, output)

    # Test that if you make a batch of signals and call it,
    # the first item in the batch is still the same as above.
    batch_size = 4
    signal = AudioSignal(audio_path, offset=10, duration=2)
    signal_batch = AudioSignal.batch([signal.clone() for _ in range(batch_size)])
    signal_batch.metadata["file_loudness"] = (
        AudioSignal(audio_path).ffmpeg_loudness().item()
    )
    kwargs = transform.instantiate(seed, signal_batch, n_params=batch_size)
    batch_output = transform(signal_batch, **kwargs)

    assert batch_output[0] == output

    ## Test that you can apply transform with the same args twice.
    signal = AudioSignal(audio_path, offset=10, duration=2)
    kwargs = transform.instantiate(seed, signal)
    output_a = transform(signal.clone(), **kwargs)
    output_b = transform(signal.clone(), **kwargs)

    assert output_a == output_b


def test_compose_basic():
    seed = 0

    audio_path = "tests/audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)
    transform = tfm.Compose(
        [
            tfm.RoomImpulseResponse(csv_files=["tests/audio/irs.csv"]),
            tfm.BackgroundNoise(csv_files=["tests/audio/noises.csv"]),
        ],
    )

    kwargs = transform.instantiate(seed, signal)
    output = transform(signal, **kwargs)

    _compare_transform("Compose", output)

    assert isinstance(transform[0], tfm.RoomImpulseResponse)
    assert isinstance(transform[1], tfm.BackgroundNoise)
    assert len(transform) == 2

    # Make sure __iter__ works
    for _tfm in transform:
        pass


class MulTransform(tfm.BaseTransform):
    def __init__(self, num):
        self.num = num
        super().__init__(keys=["num"])

    def _transform(self, signal, num):
        signal.audio_data = signal.audio_data * num[:, None, None]
        return signal

    def _instantiate(self, state):
        return {"num": self.num}


def test_compose_with_duplicate_transforms():
    muls = [0.5, 0.25, 0.125]
    transform = tfm.Compose([MulTransform(x) for x in muls])
    full_mul = np.prod(muls)

    kwargs = transform.instantiate(0)
    audio_path = "tests/audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)

    output = transform(signal.clone(), **kwargs)
    expected_output = signal.audio_data * full_mul

    assert torch.allclose(output.audio_data, expected_output)


def test_nested_compose():
    muls = [0.5, 0.25, 0.125]
    transform = tfm.Compose(
        [
            MulTransform(muls[0]),
            tfm.Compose([MulTransform(muls[1]), tfm.Compose([MulTransform(muls[2])])]),
        ]
    )
    full_mul = np.prod(muls)

    kwargs = transform.instantiate(0)
    audio_path = "tests/audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)

    output = transform(signal.clone(), **kwargs)
    expected_output = signal.audio_data * full_mul

    assert torch.allclose(output.audio_data, expected_output)


def test_sequential_compose():
    muls = [0.5, 0.25, 0.125]
    transform = tfm.Compose(
        [
            tfm.Compose([MulTransform(muls[0])]),
            tfm.Compose([MulTransform(muls[1]), MulTransform(muls[2])]),
        ]
    )
    full_mul = np.prod(muls)

    kwargs = transform.instantiate(0)
    audio_path = "tests/audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)

    output = transform(signal.clone(), **kwargs)
    expected_output = signal.audio_data * full_mul

    assert torch.allclose(output.audio_data, expected_output)


def test_choose_basic():
    seed = 0
    audio_path = "tests/audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)
    transform = tfm.Choose(
        [
            tfm.RoomImpulseResponse(csv_files=["tests/audio/irs.csv"]),
            tfm.BackgroundNoise(csv_files=["tests/audio/noises.csv"]),
        ]
    )

    kwargs = transform.instantiate(seed, signal)
    output = transform(signal.clone(), **kwargs)

    _compare_transform("Choose", output)

    transform = tfm.Choose(
        [
            MulTransform(0.0),
            MulTransform(2.0),
        ]
    )
    targets = [signal.clone() * 0.0, signal.clone() * 2.0]

    for seed in range(10):
        kwargs = transform.instantiate(seed, signal)
        output = transform(signal.clone(), **kwargs)

        assert output in targets

    # Test that if you make a batch of signals and call it,
    # the first item in the batch is still the same as above.
    batch_size = 4
    signal = AudioSignal(audio_path, offset=10, duration=2)
    signal_batch = AudioSignal.batch([signal.clone() for _ in range(batch_size)])

    kwargs = transform.instantiate(seed, signal_batch, n_params=batch_size)
    batch_output = transform(signal_batch, **kwargs)

    for nb in range(batch_size):
        assert batch_output[nb] in targets


def test_choose_weighted():
    seed = 0
    audio_path = "tests/audio/spk/f10_script4_produced.wav"
    transform = tfm.Choose(
        [
            MulTransform(0.0),
            MulTransform(2.0),
        ],
        weights=[0.0, 1.0],
    )

    # Test that if you make a batch of signals and call it,
    # the first item in the batch is still the same as above.
    batch_size = 4
    signal = AudioSignal(audio_path, offset=10, duration=2)
    signal_batch = AudioSignal.batch([signal.clone() for _ in range(batch_size)])

    targets = [signal.clone() * 0.0, signal.clone() * 2.0]

    kwargs = transform.instantiate(seed, signal_batch, n_params=batch_size)
    batch_output = transform(signal_batch, **kwargs)

    for nb in range(batch_size):
        assert batch_output[nb] == targets[1]


def test_choose_with_compose():
    audio_path = "tests/audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)

    transform = tfm.Choose(
        [
            tfm.Compose([MulTransform(0.0)]),
            tfm.Compose([MulTransform(2.0)]),
        ]
    )

    targets = [signal.clone() * 0.0, signal.clone() * 2.0]

    for seed in range(10):
        kwargs = transform.instantiate(seed, signal)
        output = transform(signal, **kwargs)

        assert output in targets


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
        signal = batch.pop("signal")
        original = signal.clone()

        signal = dataset.transform(signal, **batch)
        original = dataset.transform(original, **batch)
        mask = batch["Silence"]["mask"]

        zeros_ = torch.zeros_like(signal[mask].audio_data)
        original_ = original[~mask].audio_data

        assert torch.allclose(signal[mask].audio_data, zeros_)
        assert torch.allclose(original[~mask].audio_data, original_)


def test_nested_masking():
    transform = tfm.Compose(
        [
            tfm.VolumeNorm(prob=0.5),
            tfm.Silence(prob=0.9),
        ],
        prob=0.9,
    )

    dataset = CSVDataset(
        44100, 100, 0.5, csv_files=["tests/audio/spk.csv"], transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=0, batch_size=10, collate_fn=dataset.collate
    )

    for batch in dataloader:
        batch = util.prepare_batch(batch, device="cpu")
        signal = batch["signal"]
        kwargs = batch["transform_args"]
        with torch.no_grad():
            output = dataset.transform(signal, **kwargs)
