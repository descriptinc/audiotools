import json
import numbers
import os
import shlex
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


@dataclass
class Info:
    sample_rate: float
    num_frames: int


def info(audio_path):
    """Shim for torchaudio.info to make 0.7.2 API match 0.8.0.

    Parameters
    ----------
    audio_path : str
        Path to audio file.
    """
    info = torchaudio.info(str(audio_path))
    if isinstance(info, tuple):  # pragma: no cover
        signal_info = info[0]
        info = Info(sample_rate=signal_info.rate, num_frames=signal_info.length)
    return info


def ensure_tensor(x, ndim=None, batch_size=None):
    if isinstance(x, (float, int, numbers.Integral)):
        x = np.array([x])
    if isinstance(x, (list, tuple)):
        x = np.array(x)
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    if ndim is not None:
        if x.ndim < ndim:
            for _ in range(ndim - 1):
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
        raise ValueError(
            "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
        )


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
    except:  # pragma: no cover
        _close()
        raise
    _close()


def find_audio(folder: str, ext: List[str] = ["wav", "flac", "mp3"]):
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
        files += folder.glob(f"**/*.{x}")
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


def upload_file_to_discourse(
    path, api_username=None, api_key=None, discourse_server=None
):  # pragma: no cover
    if api_username is None:
        api_username = os.environ.get("DISCOURSE_API_USERNAME", None)
    if api_key is None:
        api_key = os.environ.get("DISCOURSE_API_KEY", None)
    if discourse_server is None:
        discourse_server = os.environ.get("DISCOURSE_SERVER", None)

    if discourse_server is None or api_key is None or api_username is None:
        raise RuntimeError(
            "DISCOURSE_API_KEY, DISCOURSE_SERVER, DISCOURSE_API_USERNAME must be set in your environment!"
        )

    command = (
        f"curl -s -X POST {discourse_server}/uploads.json "
        f"-H 'content-type: multipart/form-data;' "
        f"-H 'Api-Key: {api_key}' "
        f"-H 'Api-Username: {api_username}' "
        f"-F 'type=composer' "
        f"-F 'files[]=@{path}' "
    )
    return json.loads(subprocess.check_output(shlex.split(command)))


def upload_figure_to_discourse(
    label=None,
    fig=None,
    bbox_inches="tight",
    pad_inches=0,
    api_username=None,
    api_key=None,
    discourse_server=None,
    **kwargs,
):  # pragma: no cover
    if fig is None:
        fig = plt.gcf()

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        plt.savefig(f.name, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)

        info = upload_file_to_discourse(
            f.name,
            api_username=api_username,
            api_key=api_key,
            discourse_server=discourse_server,
        )

    if label is None:
        label = "unknown"
        formatted = f"![{label}|image]({info['short_path']})"
    return formatted, info


def audio_table(audio_dict):
    FORMAT = "| Label | Audio \n" "|---|:-: \n"

    for k, v in audio_dict.items():
        formatted_audio = v.embed(display=False, return_html=True)
        row = f"| {k} | {formatted_audio} |\n"
        FORMAT += row

    return FORMAT


def discourse_audio_table(audio_dict, **kwargs):  # pragma: no cover
    """Creates a Markdown table out of a dictionary of
    AudioSignal objects which looks something like:

    | Label | Audio
    | [key1] | [val1.audio_data embedded]
    | [key2] | [val2.audio_data embedded]

    Parameters
    ----------
    audio_dict : dict[str, AudioSignal]
        Dictionary of strings mapped to AudioSignal objects.
    """
    FORMAT = "| Label | Audio \n" "|---|:-: \n"
    uploads = []

    for k, v in audio_dict.items():
        upload = v.upload_to_discourse(k, **kwargs)
        formatted_audio = upload[0].replace("|", "\|")
        row = f"| {k} | {formatted_audio} |\n"
        FORMAT += row
        uploads.append(upload)
    return FORMAT, uploads


def disp(obj, label=None):  # pragma: no cover
    from audiotools import AudioSignal

    DISCOURSE = bool(os.environ.get("UPLOAD_TO_DISCOURSE", False))

    if isinstance(obj, AudioSignal):
        if DISCOURSE:
            info = obj.upload_to_discourse(label=label, ext=".mp3")
            print(info[0])
        else:
            audio_elem = obj.embed(display=False, return_html=True)
            print(audio_elem)
    if isinstance(obj, dict):
        if DISCOURSE:
            table = discourse_audio_table(obj, ext=".mp3")[0]
        else:
            table = audio_table(obj)
        print(table)
    if isinstance(obj, plt.Figure):
        if DISCOURSE:
            info = upload_figure_to_discourse()
            print(info[0])
        else:
            plt.show()
