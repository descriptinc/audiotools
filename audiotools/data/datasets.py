import copy
import typing
from multiprocessing import Manager
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from numpy.random import RandomState
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


def load_signal(
    path: str,
    sample_rate: int,
    num_channels: int = 1,
    state: Optional[Union[RandomState, int]] = None,
    offset: Optional[int] = None,
    duration: Optional[float] = None,
    loudness_cutoff: Optional[float] = None,
):
    if offset is None:
        signal = AudioSignal.salient_excerpt(
            path,
            duration=duration,
            state=state,
            loudness_cutoff=loudness_cutoff,
        )
    else:
        signal = AudioSignal(
            path,
            offset=offset,
            duration=duration,
        )

    if num_channels == 1:
        signal = signal.to_mono()
    signal = signal.resample(sample_rate)

    if signal.duration < duration:
        signal = signal.zero_pad_to(int(duration * sample_rate))

    return signal


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

        signal = load_signal(
            path=audio_info["path"],
            sample_rate=sample_rate,
            num_channels=num_channels,
            state=state,
            offset=offset,
            duration=duration,
            loudness_cutoff=loudness_cutoff,
        )

        for k, v in audio_info.items():
            signal.metadata[k] = v

        return signal, csv_idx


class MultiTrackAudioLoader:
    """
    This loader behaves similarly to AudioLoader, but
    it loads multiple tracks in a group, and returns a dictionary
    of AudioSignals, one for each track.

    For example, one may call this loader like this:
    ```
    loader = MultiTrackAudioLoader(
        csv_groups = [
            {
                "vocals": "datset1/vocals.csv",
                "drums": "dataset1/drums.csv",
                "bass": "dataset1/bass.csv",

                "coherence": 0.5,
                "weight": 1,
                "primary_key": "vocals",
            },
            {
                "vocals": "datset2/vocals.csv",
                "drums": "dataset2/drums.csv",
                "bass": "dataset2/bass.csv",
                "guitar": "dataset2/guitar.csv",

                "coherence": 0.5,
                "weight": 3,
                "primary_key": "vocals",
            },
        ]
    )
    ```

    There are special keys that can be passed to each csv group dictionary:

    - primary_key : str, optional
        - If provided, will load a salient excerpt from the audio file specified by the primary key.
        - If not provided, will pick the first column for each csv as the primary key.
    - weight : float, optional
        - weight for sampling this CSV group, by default 1.0
    - coherence: float, optional
        - Coherence of sampled multitrack data, by default 1.0
        - Probability of sampling a multitrack recording that is coherent.
        - A coherent multitrack recording is one the same CSV row
        is drawn for each of the sources.
        - An incoherent multitrack recording is one where a random row
        is drawn for each of the sources.

    You can change the default values for these keys by updating the
    ``MultiTrackAudioLoader.CSV_GROUP_DEFAULTS`` dictionary.

    NOTE: If no offset is provided to the loader, then the loader will
    choose a salient excerpt as dictated by the signal associated with ``primary_key``.
    This may fail if all of the signals in a given row are not of equal duration.

    Parameters
    ----------
    csv_groups: List[Dict[str, str]], optional
        List of dictionaries containing CSV files and their associated keys.
    """

    CSV_GROUP_DEFAULTS = {"csv_weight": 1.0, "coherence": 1.0}
    CSV_GROUP_RESERVED_KEYS = ["csv_weight", "coherence", "primary_key"]

    def __init__(
        self,
        csv_groups: List[Dict[str, str]] = None,
    ):

        csv_weights = [
            g.pop("csv_weight", self.CSV_GROUP_DEFAULTS["csv_weight"])
            for g in csv_groups
        ]
        csv_weights = np.exp(csv_weights) / np.sum(np.exp(csv_weights))
        self.csv_weights = csv_weights.tolist()
        self.coherences = [
            g.pop("coherence", self.CSV_GROUP_DEFAULTS["coherence"]) for g in csv_groups
        ]

        # find the set of audio columns
        # (i.e. the union of all keys across all csv groups)
        # this way, we can add zero signals for any missing tracks
        # which let's us batch different csv groups together
        csv_group_keys = [list(g.keys()) for g in csv_groups]
        self.audio_columns = list(
            set(
                [
                    key
                    for keys in csv_group_keys
                    for key in keys
                    if key not in self.CSV_GROUP_RESERVED_KEYS
                ]
            )
        )
        self.primary_keys = [
            g.pop("primary_key", keys[0]) for g, keys in zip(csv_groups, csv_group_keys)
        ]

        self.csv_groups = csv_groups
        self.audio_lists = []
        for csv_dict in csv_groups:
            self.audio_lists.append(
                {
                    k: util.read_csv([v], remove_empty=False)[0]
                    for k, v in csv_dict.items()
                }
            )

        # default to the first column of each csv as the primary key
        if self.primary_keys is None:
            self.primary_keys = [list(csv_dict.keys())[0] for csv_dict in csv_groups]

        for key, csv_group in zip(self.primary_keys, self.csv_groups):
            if key not in csv_group.keys():
                raise ValueError(
                    f"Primary key {key} not found in csv keys {csv_group.keys()}"
                )

    def __call__(
        self,
        state,
        sample_rate: int,
        duration: float,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        offset: float = None,
    ):
        # pick a group of csvs
        csv_group_idx = state.choice(len(self.audio_lists), p=self.csv_weights)

        # grab the group of csvs and primary key for this group
        csv_group = self.audio_lists[csv_group_idx]
        primary_key = self.primary_keys[csv_group_idx]

        # if not coherent, sample the csv idxs for each track independently
        coherence = self.coherences[csv_group_idx]
        coherent = state.rand() < coherence
        if not coherent:
            csv_idxs = state.choice(
                len(csv_group[primary_key]), size=len(csv_group), replace=False
            )
            csv_idxs = {key: csv_idxs[i] for i, key in enumerate(csv_group.keys())}
        else:
            # otherwise, use the same csv idx for each track
            choice_idx = state.choice(len(csv_group[primary_key]))
            csv_idxs = {key: choice_idx for key in csv_group.keys()}

        # pick a row from the primary csv
        csv_idx = csv_idxs[primary_key]
        p_audio_info = csv_group[primary_key][csv_idx]

        # load the primary signal first (if it exists in this row),
        # and use it to determine the offset and duration
        if p_audio_info["path"] == "":
            primary_signal = AudioSignal.zeros(
                sample_rate=sample_rate, num_channels=num_channels, duration=duration
            )
        else:
            primary_signal = load_signal(
                path=p_audio_info["path"],
                sample_rate=sample_rate,
                num_channels=num_channels,
                state=state,
                offset=offset,
                duration=duration,
                loudness_cutoff=loudness_cutoff,
            )
            for k, v in p_audio_info.items():
                primary_signal.metadata[k] = v

            # update the offset and duration according to the primary signal
            offset = primary_signal.metadata["offset"]
            duration = primary_signal.metadata["duration"]

        # load the rest of the signals
        signals = {}
        for audio_key, audio_list in csv_group.items():
            if audio_key == primary_key:
                signals[audio_key] = primary_signal
                continue

            csv_idx = csv_idxs[audio_key]
            audio_info = audio_list[csv_idx]

            # if the path is empty, then skip
            # and add a zero signal later
            if audio_info["path"] == "":
                continue

            signal = load_signal(
                path=audio_info["path"],
                sample_rate=sample_rate,
                num_channels=num_channels,
                state=state,
                offset=offset,
                duration=duration,
                loudness_cutoff=loudness_cutoff,
            )
            for k, v in audio_info.items():
                signal.metadata[k] = v

            signals[audio_key] = signal

        # add zero signals for any missing tracks
        for k in self.audio_columns:
            if k not in signals:
                signals[k] = AudioSignal.zeros(
                    duration=duration,
                    num_channels=num_channels,
                    sample_rate=sample_rate,
                )

        for signal in signals.values():
            assert signal.duration == duration

        return signals, csv_group_idx


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


class CSVMultiTrackDataset(BaseDataset):
    """
    A dataset for loading coherent multitrack data for source separation.

    This dataset behaves similarly to CSV dataset, but instead of
    passing a list of single CSV files, you must pass a list of
    dictionaries of CSV files, where each dictionary represents
    a group of multitrack files that should be loaded together.

    NOTE: Within a dictionary, each CSV file must have the same
    number of rows, since it is expected that each row represents
    a single track in a multitrack recording.

    For example, our list of CSV groups might look like this:

    ```
    csv_groups = [
        {
            "vocals": "dataset1/vocals.csv",
            "drums": "dataset1/drums.csv",
            "bass": "dataset1/bass.csv",
            "coherence_prob": 0.5,          # probability of sampling coherent multitracks.
            "primary_key": "vocals",        # the key of the primary track.
            "weight": 1.0,                  # the weight for sampling this group
        },
        {
            "vocals": "datset2/vocals.csv",
            "drums": "dataset2/drums.csv",
            "bass": "dataset2/bass.csv",
            "guitar": "dataset2/guitar.csv",
            "coherence_prob": 1.0,
            "primary_key": "vocals",
            "weight": 1.0,
        },
    ]
    ```

    There are special keys that can be passed to each csv group dictionary:

    - primary_key : str, optional
        - If provided, will load a salient excerpt from the audio file specified by the primary key.
        - If not provided, will pick the first column for each csv as the primary key.
    - weight : float, optional
        - weight for sampling this CSV group, by default 1.0
    - coherence: float, optional
        - Coherence of sampled multitrack data, by default 1.0
        - Probability of sampling a multitrack recording that is coherent.
        - A coherent multitrack recording is one the same CSV row
        is drawn for each of the sources.
        - An incoherent multitrack recording is one where a random row
        is drawn for each of the sources.

    You can change the default values for these keys by updating the
    ``MultiTrackAudioLoader.CSV_GROUP_DEFAULTS`` dictionary.

    You can create a multitrack dataset that behaves similar to
    a regular CSV dataset:

    ```
    import audiotools

    transform = audiotools.transforms.Identity()

    # csv dataset
    csv_dataset = audiotools.data.datasets.CSVDataset(
        44100,
        n_examples=100,
        csv_files=["tests/audio/spk.csv"],
        transform=transform,
    )
    # get an item
    data = csv_dataset[0]
    # get the signal
    signal = data["signal"]
    print(signal)

    # multitrack dataset
    multitrack_dataset = audiotools.data.datasets.CSVMultiTrackDataset(
        44100,
        n_examples=100,
        csv_groups=[{
            "speaker": "tests/audio/spk.csv"
        }],
        transform={
            "speaker": transform,
        }
    )

    # take an item from the dataset
    data = multitrack_dataset[0]

    # access the audio signal
    signal = data["signals"]["speaker"]
    print(signal)
    ```

    Parameters
    ----------
    sample_rate : int
        Sample rate of audio.
    csv_groups: List[Dict[str, str]], optional
        List of dictionaries containing CSV files and their associated keys.
    n_examples : int, optional
        Number of examples, by default 1000
    duration : float, optional
        Duration of excerpts, in seconds, by default 0.5
    num_channels : int, optional
        Number of channels, by default 1
    transforms : Dict[str, typing.Callable], optional
        Dict of transforms, one for each source.

    Usage
    -----
    ```python
    source_transforms = {
        "vocals": audiotools.transforms.Identity(),
        "drums": audiotools.transforms.Identity(),
        "bass": audiotools.transforms.Identity(),
    }
    dataset = CSVMultiTrackDataset(
        sample_rate=44100,
        n_examples=20,
        csv_groups=[{
            "drums": "tests/audio/musdb-7s/drums.csv",
            "bass": "tests/audio/musdb-7s/bass.csv",
            "vocals": "tests/audio/musdb-7s/vocals.csv",

            "coherence_prob": 0.95,
            "primary_key": "drums",
        }, {
            "drums": "tests/audio/musdb-7s/drums.csv",
            "bass": "tests/audio/musdb-7s/bass.csv",

            "coherence_prob": 1.0,
            "primary_key": "drums",
        }],
        transform=source_transforms,
    )

    assert set(dataset.source_names) == set(["bass", "drums", "vocals"])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        num_workers=0,
        collate_fn=dataset.collate,
    )

    for batch in dataloader:
        kwargs = batch["transform_args"]
        signals = batch["signals"]

        tfmed = {
            k: dataset.transform[k](sig.clone(), **kwargs[k])
            for k, sig in signals.items()
        }
        mix = sum(tfmed.values())
    ```
    """

    def __init__(
        self,
        sample_rate: int,
        n_examples: int = 1000,
        duration: float = 0.5,
        csv_groups: List[Dict[str, str]] = None,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        transform: Dict[str, Callable] = None,
    ):
        self.loader = MultiTrackAudioLoader(csv_groups)

        self.num_channels = num_channels
        self.loudness_cutoff = loudness_cutoff

        if transform is None:
            transform = {}

        assert isinstance(
            transform, dict
        ), "transform for CSVMultiTrackDataset must be a dict"
        for key in self.loader.audio_columns:
            if key not in transform:
                from .transforms import Identity

                transform[key] = Identity()

        super().__init__(
            n_examples, duration=duration, transform=transform, sample_rate=sample_rate
        )

    @property
    def source_names(self):
        return list(self.loader.audio_columns)

    @property
    def primary_keys(self):
        return self.loader.primary_keys

    def __getitem__(self, idx):
        state = util.random_state(idx)

        signals, csv_idx = self.loader(
            state,
            self.sample_rate,
            duration=self.duration,
            loudness_cutoff=self.loudness_cutoff,
            num_channels=self.num_channels,
        )

        # Instantiate the transform.
        transform_kwargs = {
            k: self.transform[k].instantiate(state, signal=signals[k]) for k in signals
        }

        item = {
            "idx": idx,
            "signals": signals,
            "label": csv_idx,
            "transform_args": transform_kwargs,
        }
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
