import os
import random
import tempfile

import numpy as np
import pytest
import torch

from audiotools import util
from audiotools.core.audio_signal import AudioSignal


def test_check_random_state():
    # seed is None
    rng_type = type(np.random.RandomState(10))
    rng = util.random_state(None)
    assert type(rng) == rng_type

    # seed is int
    rng = util.random_state(10)
    assert type(rng) == rng_type

    # seed is RandomState
    rng_test = np.random.RandomState(10)
    rng = util.random_state(rng_test)
    assert type(rng) == rng_type

    # seed is none of the above : error
    pytest.raises(ValueError, util.random_state, "random")


def test_seed():
    util.seed(0)
    torch_result_a = torch.randn(1)
    np_result_a = np.random.randn(1)
    py_result_a = random.random()

    util.seed(0, set_cudnn=True)
    torch_result_b = torch.randn(1)
    np_result_b = np.random.randn(1)
    py_result_b = random.random()

    assert torch_result_a == torch_result_b
    assert np_result_a == np_result_b
    assert py_result_a == py_result_b


def test_hz_to_bin():
    hz = torch.from_numpy(np.array([100, 200, 300]))
    sr = 1000
    n_fft = 2048

    bins = util.hz_to_bin(hz, n_fft, sr)

    assert (((bins / n_fft) * sr) - hz).abs().max() < 1


def test_find_audio():
    audio_files = util.find_audio("tests/", ["wav"])
    for a in audio_files:
        assert "wav" in str(a)

    audio_files = util.find_audio("tests/", ["flac"])
    assert not audio_files


def test_chdir():
    with tempfile.TemporaryDirectory(suffix="tmp") as d:
        with util.chdir(d):
            assert os.path.samefile(d, os.path.realpath("."))


def test_prepare_batch():
    batch = {"tensor": torch.randn(1), "non_tensor": np.random.randn(1)}
    util.prepare_batch(batch)

    batch = torch.randn(1)
    util.prepare_batch(batch)

    batch = [torch.randn(1), np.random.randn(1)]
    util.prepare_batch(batch)


def test_sample_dist():
    state = util.random_state(0)
    v1 = state.uniform(0.0, 1.0)
    v2 = util.sample_from_dist(("uniform", 0.0, 1.0), 0)
    assert v1 == v2

    assert util.sample_from_dist(("const", 1.0)) == 1.0

    dist_tuple = ("choice", [8, 16, 32])
    assert util.sample_from_dist(dist_tuple) in [8, 16, 32]


def test_collate():
    batch_size = 16

    def _one_item():
        return {
            "signal": AudioSignal(torch.randn(1, 1, 44100), 44100),
            "tensor": torch.randn(1),
            "string": "Testing",
            "dict": {
                "nested_signal": AudioSignal(torch.randn(1, 1, 44100), 44100),
            },
        }

    items = [_one_item() for _ in range(batch_size)]
    collated = util.collate(items)

    assert collated["signal"].batch_size == batch_size
    assert collated["tensor"].shape[0] == batch_size
    assert len(collated["string"]) == batch_size
    assert collated["dict"]["nested_signal"].batch_size == batch_size
