import typing
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import numpy as np
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from ..core import AudioSignal
from ..core import util


class AudioLoader:
    """Loads audio endlessly from a list of audio sources
    containing paths to audio files. Audio sources can be
    folders full of audio files (which are found via file
    extension) or by providing a CSV file which contains paths
    to audio files.

    Parameters
    ----------
    sources : List[str], optional
        Sources containing folders, or CSVs with
        paths to audio files, by default None
    weights : List[float], optional
        Weights to sample audio files from each source, by default None
    relative_path : str, optional
        Path audio should be loaded relative to, by default ""
    transform : Callable, optional
        Transform to instantiate and apply to loaded audio,
        by default None
    """

    def __init__(
        self,
        sources: List[str] = None,
        weights: List[float] = None,
        transform: Callable = None,
        relative_path: str = "",
        ext: List[str] = util.AUDIO_EXTENSIONS,
    ):
        self.audio_lists = util.read_sources(
            sources, relative_path=relative_path, ext=ext
        )
        self.sources = sources
        self.weights = weights
        self.transform = transform

    def __call__(
        self,
        state,
        sample_rate: int,
        duration: float,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        offset: float = None,
        source_idx: int = None,
        item_idx: int = None,
    ):
        if source_idx is not None and item_idx is not None:
            try:
                audio_info = self.audio_lists[source_idx][item_idx]
            except:
                audio_info = {"path": None}
        else:
            audio_info, source_idx, item_idx = util.choose_from_list_of_lists(
                state, self.audio_lists, p=self.weights
            )

        path = audio_info["path"]
        signal = AudioSignal.zeros(duration, sample_rate, num_channels)

        if path is not None:
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
        else:
            path = "n/a"

        if num_channels == 1:
            signal = signal.to_mono()
        signal = signal.resample(sample_rate)

        if signal.duration < duration:
            signal = signal.zero_pad_to(int(duration * sample_rate))

        for k, v in audio_info.items():
            signal.metadata[k] = v

        item = {
            "signal": signal,
            "source_idx": source_idx,
            "item_idx": item_idx,
            "source": self.sources[source_idx],
            "path": path,
        }
        if self.transform is not None:
            item["transform_args"] = self.transform.instantiate(state, signal=signal)
        return item


class AudioDataset:
    """_summary_

    Parameters
    ----------
    loaders : Union[AudioLoader, List[AudioLoader], Dict[str, AudioLoader]]
        _description_
    sample_rate : int
        _description_
    n_examples : int, optional
        _description_, by default 1000
    duration : float, optional
        _description_, by default 0.5
    loudness_cutoff : float, optional
        _description_, by default -40
    num_channels : int, optional
        _description_, by default 1
    transform : Callable, optional
        _description_, by default None
    aligned : bool, optional
        _description_, by default False
    shuffle_loaders : bool, optional
        _description_, by default False
    """

    def __init__(
        self,
        loaders: Union[AudioLoader, List[AudioLoader], Dict[str, AudioLoader]],
        sample_rate: int,
        n_examples: int = 1000,
        duration: float = 0.5,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        transform: Callable = None,
        aligned: bool = False,
        shuffle_loaders: bool = False,
    ):
        # Internally we convert loaders to a dictionary
        if isinstance(loaders, list):
            loaders = {i: l for i, l in enumerate(loaders)}
        elif isinstance(loaders, AudioLoader):
            loaders = {0: loaders}

        self.loaders = loaders
        self.loudness_cutoff = loudness_cutoff
        self.num_channels = num_channels

        self.length = n_examples
        self.transform = transform
        self.sample_rate = sample_rate
        self.duration = duration
        self.aligned = aligned
        self.shuffle_loaders = shuffle_loaders

    def __getitem__(self, idx):
        state = util.random_state(idx)

        source_idx = None
        item_idx = None
        offset = None
        item = {}

        keys = list(self.loaders.keys())
        if self.shuffle_loaders:
            state.shuffle(keys)

        for key in keys:
            loader = self.loaders[key]
            item[key] = loader(
                state,
                self.sample_rate,
                offset=offset,
                duration=self.duration,
                loudness_cutoff=self.loudness_cutoff,
                num_channels=self.num_channels,
                source_idx=source_idx,
                item_idx=item_idx,
            )

            if self.aligned:
                source_idx = item[key]["source_idx"]
                item_idx = item[key]["item_idx"]
                offset = item[key]["signal"].metadata["offset"]

        # Sort dictionary back into original order
        keys = list(self.loaders.keys())
        item = {k: item[k] for k in keys}

        item["idx"] = idx
        if self.transform is not None:
            item["transform_args"] = self.transform.instantiate(
                state=state, signal=item[keys[0]]["signal"]
            )

        # If there's only one loader, pop it up
        # to the main dictionary, instead of keeping it
        # nested.
        if len(keys) == 1:
            item.update(item.pop(keys[0]))

        return item

    def __len__(self):
        return self.length

    @staticmethod
    def collate(list_of_dicts: Union[list, dict]):
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
