import numpy as np
from audiotools import util
import pytest
import torch
import tempfile
import os
from pathlib import Path

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
    pytest.raises(ValueError, util.random_state, 'random')


def test_hz_to_bin():
    hz = torch.from_numpy(np.array([100, 200, 300]))
    sr = 1000
    n_fft = 2048

    bins = util.hz_to_bin(hz, n_fft, sr)

    assert (((bins / n_fft) * sr) - hz).abs().max() < 1

def test_find_audio():
    audio_files = util.find_audio('tests/', ['wav'])
    for a in audio_files:
        assert 'wav' in str(a)

    audio_files = util.find_audio('tests/', ['flac'])
    assert not audio_files

def test_chdir():
    with tempfile.TemporaryDirectory(suffix='tmp') as d:
        with util.chdir(d):
            assert os.path.samefile(d, os.path.realpath('.'))
