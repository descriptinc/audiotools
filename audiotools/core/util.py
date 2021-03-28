from contextlib import contextmanager
import os
from functools import wraps
import numpy as np
import numbers
import torch
from typing import List
from pathlib import Path

def ensure_tensor(x, ndim=None, batch_size=None):
    if isinstance(x, (float, int, numbers.Integral)):
        x = np.array([x])
    if isinstance(x, (list, tuple)):
        x = np.array(x)
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    if ndim is not None:
        if x.ndim < ndim:
            for _ in range(ndim-1):
                x = x.unsqueeze(-1) 
    if batch_size is not None:
        if x.shape[0] != batch_size:
            shape = list(x.shape)
            shape[0] = batch_size
            x = x.expand(*shape)
    x = x.float()
    return x

def _get_value(other):
    from . import AudioSignal
    if isinstance(other, AudioSignal):
        return other.audio_data
    return other

def hz_to_bin(hz, n_fft, sample_rate):
    shape = hz.shape
    hz = hz.flatten()
    freqs = torch.linspace(0, sample_rate / 2, 2 + n_fft // 2)
    hz[hz > sample_rate / 2] = sample_rate / 2

    closest = (hz[None, :] - freqs[:, None]).abs()
    closest_bins = closest.min(dim=0).indices

    return closest_bins.reshape(*shape)

def random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    elif isinstance(seed, (numbers.Integral, np.integer, int)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                         ' instance' % seed)

@contextmanager
def _close_temp_files(tmpfiles):
    """
    Utility function for creating a context and closing all temporary files
    once the context is exited. For correct functionality, all temporary file
    handles created inside the context must be appended to the ```tmpfiles```
    list.

    This function is taken wholesale from Scaper.

    Args:
        tmpfiles (list): List of temporary file handles
    """
    def _close():
        for t in tmpfiles:
            try:
                t.close()
                os.unlink(t.name)
            except:
                pass
    try:
        yield
    except: # pragma: no cover
        _close()
        raise
    _close()

def find_audio(
    folder : str, 
    ext : List[str] = ['wav', 'flac', 'mp3']
):
    """
    Finds all audio files in a directory 
    recursively. Returns a list.
    Parameters
    ----------
    folder : str
        Folder to look for audio files in, recursively.
    ext : List[str], optional
        Extensions to look for without the ., by default
        ['wav', 'flac', 'mp3'].
    """
    folder = Path(folder)        
    files = []
    for x in ext:
        files += folder.glob(f'**/*.{x}')
    return files

@contextmanager
def chdir(newdir):
    """
    Context manager for switching directories to run a 
    function. Useful for when you want to use relative
    paths to different runs.
    Parameters
    ----------
    newdir : str
        Directory to switch to.
    """
    curdir = os.getcwd()
    try:
        os.chdir(newdir)
        yield
    finally:
        os.chdir(curdir)
