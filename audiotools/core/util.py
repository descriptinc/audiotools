import csv
import numbers
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from flatten_dict import flatten
from flatten_dict import unflatten
from librosa import util


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
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if ndim is not None:
        assert x.ndim <= ndim
        while x.ndim < ndim:
            x = x.unsqueeze(-1)
    if batch_size is not None:
        if x.shape[0] != batch_size:
            shape = list(x.shape)
            shape[0] = batch_size
            x = x.expand(*shape)
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


def seed(random_seed, set_cudnn=False):
    """
    Seeds all random states with the same random seed
    for reproducibility. Seeds ``numpy``, ``random`` and ``torch``
    random generators.
    For full reproducibility, two further options must be set
    according to the torch documentation:
    https://pytorch.org/docs/stable/notes/randomness.html
    To do this, ``set_cudnn`` must be True. It defaults to
    False, since setting it to True results in a performance
    hit.

    Args:
        random_seed (int): integer corresponding to random seed to
        use.
        set_cudnn (bool): Whether or not to set cudnn into determinstic
        mode and off of benchmark mode. Defaults to False.
    """

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if set_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


AUDIO_EXTENSIONS = ["wav", "flac", "mp3", "mp4"]


def find_audio(folder: str, ext: List[str] = AUDIO_EXTENSIONS):
    """
    Finds all audio files in a directory
    recursively. Returns a list.
    Parameters
    ----------
    folder : str
        Folder to look for audio files in, recursively.
    ext : List[str], optional
        Extensions to look for without the ., by default
        ['wav', 'flac', 'mp3', 'mp4'].
    """
    folder = Path(folder)
    files = []
    for x in ext:
        files += folder.glob(f"**/*.{x}")
    return files


def read_csv(filelists):
    files = []
    data_path = Path(os.getenv("PATH_TO_DATA", ""))
    for filelist in filelists:
        with open(filelist, "r") as f:
            reader = csv.DictReader(f)
            _files = []
            for x in reader:
                x["path"] = str(data_path / x["path"])
                _files.append(x)
        files.append(sorted(_files, key=lambda x: x["path"]))
    return files


def choose_from_list_of_lists(state, list_of_lists, p=None):
    idx = state.choice(list(range(len(list_of_lists))), p=p)
    item_idx = state.randint(len(list_of_lists[idx]))
    return list_of_lists[idx][item_idx], idx


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


def prepare_batch(batch, device="cpu"):
    if isinstance(batch, dict):
        batch = flatten(batch)
        for key, val in batch.items():
            try:
                batch[key] = val.to(device)
            except:
                pass
        batch = unflatten(batch)
    elif torch.is_tensor(batch):
        batch = batch.to(device)
    elif isinstance(batch, list):
        for i in range(len(batch)):
            try:
                batch[i] = batch[i].to(device)
            except:
                pass
    return batch


def sample_from_dist(dist_tuple, state=None):
    if dist_tuple[0] == "const":
        return dist_tuple[1]
    state = random_state(state)
    dist_fn = getattr(state, dist_tuple[0])
    return dist_fn(*dist_tuple[1:])


def collate(list_of_dicts):
    from . import AudioSignal

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


BASE_SIZE = 864
DEFAULT_FIG_SIZE = (9, 3)


def format_figure(
    fig_size: tuple = None,
    title: str = None,
    fig=None,
    format_axes: bool = True,
    format: bool = True,
    font_color: str = "white",
):
    if fig_size is None:
        fig_size = DEFAULT_FIG_SIZE
    if not format:
        return
    if fig is None:
        fig = plt.gcf()
    fig.set_size_inches(*fig_size)
    axs = fig.axes

    pixels = (fig.get_size_inches() * fig.dpi)[0]
    font_scale = pixels / BASE_SIZE

    if format_axes:
        axs = fig.axes

        for ax in axs:
            ymin, _ = ax.get_ylim()
            xmin, _ = ax.get_xlim()

            ticks = ax.get_yticks()
            for t in ticks[2:-1]:
                t = axs[0].annotate(
                    f"{(t / 1000):2.1f}k",
                    xy=(xmin, t),
                    xycoords="data",
                    xytext=(5, -5),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    color=font_color,
                    fontsize=12 * font_scale,
                    alpha=0.75,
                )

            ticks = ax.get_xticks()[2:]
            for t in ticks[:-1]:
                t = axs[0].annotate(
                    f"{t:2.1f}s",
                    xy=(t, ymin),
                    xycoords="data",
                    xytext=(5, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color=font_color,
                    fontsize=12 * font_scale,
                    alpha=0.75,
                )

            ax.margins(0, 0)
            ax.set_axis_off()
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    if title is not None:
        t = axs[0].annotate(
            title,
            xy=(1, 1),
            xycoords="axes fraction",
            fontsize=20 * font_scale,
            xytext=(-5, -5),
            textcoords="offset points",
            ha="right",
            va="top",
            color="white",
        )
        t.set_bbox(dict(facecolor="black", alpha=0.5, edgecolor="black"))
