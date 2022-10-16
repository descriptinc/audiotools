import copy
import typing
from multiprocessing import Manager
from typing import List

from torch.utils.data import BatchSampler as _BatchSampler
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from ..core import AudioSignal
from ..core import util

# We need to set SHARED_KEYS statically, with no relationship to the
# BaseDataset object, or we'll hit RecursionErrors in the lookup.
SHARED_KEYS = [
    "signal",
    "duration",
    "shared_transform",
    "check_transform",
    "sample_rate",
    "batch_size",
]


class SharedMixin:
    """Mixin which creates a set of keys that are shared across processes.

    The getter looks up the name in ``SHARED_KEYS`` (see above). If it's there,
    return it from the dictionary that is kept in shared memory.
    Otherwise, do the normal ``__getattribute__``. This line only
    runs if the key is in ``SHARED_KEYS``.

    The setter looks up the name in ``SHARED_KEYS``. If it's there
    set the value in the dictionary accordingly, so that it the other
    dataset replicas know about it. Otherwise, do the normal
    ``__setattr__``. This line only runs if the key is in ``SHARED_KEYS``.

    >>> SHARED_KEYS = [
    >>>     "signal",
    >>>     "duration",
    >>>     "shared_transform",
    >>>     "check_transform",
    >>>     "sample_rate",
    >>>     "batch_size",
    >>> ]

    """

    def __getattribute__(self, name: str):
        if name in SHARED_KEYS:
            return self.shared_dict[name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in SHARED_KEYS:
            self.shared_dict[name] = value
        else:
            super().__setattr__(name, value)


class BaseDataset(SharedMixin):
    """This BaseDataset class adds all the necessary logic so that there is
    a dictionary that is shared across processes when working with a
    DataLoader with num_workers > 0.

    It adds an attribute called ``shared_dict``, and changes the
    ``getattr` and ``setattr`` for the object so that it looks things up
    in the shared_dict if it's in the above ``SHARED_KEYS``. The complexity
    here is coming from working around a few quirks in multiprocessing.

    Parameters
    ----------
    length : int
        Length of the dataset.
    transform : typing.Callable, optional
        Transform to instantiate and apply to every item , by default None
    """

    def __init__(self, length: int, transform: typing.Callable = None, **kwargs):
        super().__init__()
        self.length = length
        # The following snippet of code is how we share a
        # parameter across workers in a DataLoader, without
        # introducing syntax overhead upstream.

        # 1. We use a Manager object, which is shared between
        # dataset replicas that are passed to the workers.
        self.shared_dict = Manager().dict()

        # Instead of setting `self.duration = duration` for example, we
        # instead first set it inside the `self.shared_dict` object. Further
        # down, we'll make it so that `self.duration` still works, but
        # it works by looking up the key "duration" in `self.shared_dict`.
        for k, v in kwargs.items():
            if k in SHARED_KEYS:
                self.shared_dict[k] = v

        self.shared_dict["shared_transform"] = copy.deepcopy(transform)
        self.shared_dict["check_transform"] = False
        self._transform = transform
        self.length = length

    @property
    def transform(self):
        """Transform that is associated with the dataset, copied from
        the shared dictionary so that it's up to date, but executution of
        "instantiate" will be done within each worker.
        """
        if self.check_transform:
            self._transform = copy.deepcopy(self.shared_transform)
            self.check_transform = False
        return self._transform

    @transform.setter
    def transform(self, value):
        self.shared_transform = value
        self.check_transform = True

    def __len__(self):
        return self.length

    @staticmethod
    def collate(list_of_dicts: typing.Union[list, dict]):
        """Collates items drawn from this dataset. Uses
        :py:func:`audiotools.core.util.collate`.

        Parameters
        ----------
        list_of_dicts : typing.Union[list, dict]
            Data drawn from each item.

        Returns
        -------
        dict
            Dictionary of batched data.
        """
        return util.collate(list_of_dicts)


class AudioLoader:
    """Loads audio endlessly from a list of CSV files
    containing paths to audio files.

    Parameters
    ----------
    csv_files : List[str], optional
        CSV files containing paths to audio files, by default None
    csv_weights : List[float], optional
        Weights to sample audio files from each CSV, by default None
    """

    def __init__(
        self,
        csv_files: List[str] = None,
        csv_weights: List[float] = None,
    ):
        self.audio_lists = util.read_csv(csv_files)
        self.csv_weights = csv_weights

    def __call__(
        self,
        state,
        sample_rate: int,
        duration: float,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        offset: float = None,
    ):
        audio_info, csv_idx = util.choose_from_list_of_lists(
            state, self.audio_lists, p=self.csv_weights
        )

        if offset is None:
            signal = AudioSignal.salient_excerpt(
                audio_info["path"],
                duration=duration,
                state=state,
                loudness_cutoff=loudness_cutoff,
            )
        else:
            signal = AudioSignal(
                audio_info["path"],
                offset=offset,
                duration=duration,
            )
        for k, v in audio_info.items():
            signal.metadata[k] = v

        if num_channels == 1:
            signal = signal.to_mono()
        signal = signal.resample(sample_rate)
        return signal, csv_idx


class CSVDataset(BaseDataset):
    """This is the core data handling routine in this library.
    It expects to draw ``n_examples`` audio files at a specified
    ``sample_rate`` of a specified ``duration`` from a list
    of ``csv_files`` with probability of each file being
    given by ``csv_weights``. All excerpts drawn
    will be above the specified ``loudness_cutoff``, have the
    same ``num_channels``. A transform is also instantiated using
    the index of the item, which is used to actually apply the
    transform to the item.

    Parameters
    ----------
    sample_rate : int
        Sample rate of audio.
    n_examples : int, optional
        Number of examples, by default 1000
    duration : float, optional
        Duration of excerpts, in seconds, by default 0.5
    csv_files : List[str], optional
        List of CSV files, by default None
    csv_weights : List[float], optional
        List of weights of CSV files, by default None
    loudness_cutoff : float, optional
        Loudness cutoff in decibels, by default -40
    num_channels : int, optional
        Number of channels, by default 1
    transform : typing.Callable, optional
        Transform to instantiate with each item, by default None

    Examples
    --------

    >>> transform = tfm.Compose(
    >>>     [
    >>>         tfm.VolumeNorm(),
    >>>         tfm.Silence(prob=0.5),
    >>>     ],
    >>> )
    >>> dataset = audiotools.data.datasets.CSVDataset(
    >>>     44100,
    >>>     n_examples=100,
    >>>     csv_files=["tests/audio/spk.csv"],
    >>>     transform=transform,
    >>> )
    >>> dataloader = torch.utils.data.DataLoader(
    >>>     dataset,
    >>>     batch_size=16,
    >>>     num_workers=0,
    >>>     collate_fn=dataset.collate,
    >>> )
    >>>
    >>>
    >>> for batch in dataloader:
    >>>     kwargs = batch["transform_args"]
    >>>     signal = batch["signal"]
    >>>     original = signal.clone()
    >>>
    >>>     signal = dataset.transform(signal, **kwargs)
    >>>     original = dataset.transform(original, **kwargs)
    >>>
    >>>     mask = kwargs["Compose"]["1.Silence"]["mask"]
    >>>
    >>>     zeros_ = torch.zeros_like(signal[mask].audio_data)
    >>>     original_ = original[~mask].audio_data
    >>>
    >>>     assert torch.allclose(signal[mask].audio_data, zeros_)
    >>>     assert torch.allclose(signal[~mask].audio_data, original_)

    """

    def __init__(
        self,
        sample_rate: int,
        n_examples: int = 1000,
        duration: float = 0.5,
        csv_files: List[str] = None,
        csv_weights: List[float] = None,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        transform: typing.Callable = None,
    ):
        super().__init__(
            n_examples, duration=duration, transform=transform, sample_rate=sample_rate
        )

        self.loader = AudioLoader(csv_files, csv_weights)
        self.loudness_cutoff = loudness_cutoff
        self.num_channels = num_channels

    def __getitem__(self, idx):
        state = util.random_state(idx)

        signal, csv_idx = self.loader(
            state,
            self.sample_rate,
            duration=self.duration,
            loudness_cutoff=self.loudness_cutoff,
            num_channels=self.num_channels,
        )

        # Instantiate the transform.
        item = {
            "idx": idx,
            "signal": signal,
            "label": csv_idx,
        }
        if self.transform is not None:
            item["transform_args"] = self.transform.instantiate(state, signal=signal)
        return item


# Samplers
class BatchSampler(_BatchSampler, SharedMixin):
    """BatchSampler that is like the default batch sampler, but shares
    the batch size across each worker, so that batch size can be
    manipulated across all workers on the fly during training."""

    def __init__(self, sampler, batch_size: int, drop_last: bool = False):
        self.shared_dict = Manager().dict()
        super().__init__(sampler, batch_size, drop_last=drop_last)


class ResumableDistributedSampler(DistributedSampler):  # pragma: no cover
    """Distributed sampler that can be resumed from a given start index."""

    def __init__(self, dataset, start_idx: int = None, **kwargs):
        super().__init__(dataset, **kwargs)
        # Start index, allows to resume an experiment at the index it was
        self.start_idx = start_idx // self.num_replicas if start_idx is not None else 0

    def __iter__(self):
        for i, idx in enumerate(super().__iter__()):
            if i >= self.start_idx:
                yield idx
        self.start_idx = 0  # set the index back to 0 so for the next epoch


class ResumableSequentialSampler(SequentialSampler):  # pragma: no cover
    """Sequential sampler that can be resumed from a given start index."""

    def __init__(self, dataset, start_idx: int = None, **kwargs):
        super().__init__(dataset, **kwargs)
        # Start index, allows to resume an experiment at the index it was
        self.start_idx = start_idx if start_idx is not None else 0

    def __iter__(self):
        for i, idx in enumerate(super().__iter__()):
            if i >= self.start_idx:
                yield idx
        self.start_idx = 0  # set the index back to 0 so for the next epoch
