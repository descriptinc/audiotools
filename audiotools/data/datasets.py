import copy
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
    def __getattribute__(self, name: str):
        # Look up the name in SHARED_KEYS (see above). If it's there,
        # return it from the dictionary that is kept in shared memory.
        # Otherwise, do the normal __getattribute__. This line only
        # runs if the key is in SHARED_KEYS.
        if name in SHARED_KEYS:
            return self.shared_dict[name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        # Look up the name in SHARED_KEYS (see above). If it's there
        # set the value in the dictionary accordingly, so that it the other
        # dataset replicas know about it. Otherwise, do the normal
        # __setattr__. This line only runs if the key is in SHARED_KEYS.
        if name in SHARED_KEYS:
            self.shared_dict[name] = value
        else:
            super().__setattr__(name, value)


class BaseDataset(SharedMixin):
    """This BaseDataset class adds all the necessary logic so that there is
    a dictionary that is shared across processes when working with a
    DataLoader with num_workers > 0. It adds an attribute called
    `shared_dict`, and changes the getattr and setattr for the object
    so that it looks things up in the shared_dict if it's in the above
    SHARED_KEYS. The complexity here is coming from working around a few
    quirks in multiprocessing.
    """

    def __init__(self, length, transform=None, **kwargs):
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
        # Copy the transform from the shared dict, so that it's
        # up to date, but execution of "instantiate" will be
        # done within each worker.
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
    def collate(list_of_dicts):
        return util.collate(list_of_dicts)


class AudioLoader:
    def __init__(
        self,
        csv_files: List[str] = None,
        csv_weights: List[float] = None,
        loudness_cutoff: float = -40,
    ):
        self.audio_lists = util.read_csv(csv_files)
        self.csv_weights = csv_weights
        self.loudness_cutoff = loudness_cutoff

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
    def __init__(
        self,
        sample_rate: int,
        n_examples: int = 1000,
        duration: float = 0.5,
        csv_files: List[str] = None,
        csv_weights: List[float] = None,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        transform=None,
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
            "metadata": signal.metadata,
        }
        if self.transform is not None:
            item["transform_args"] = self.transform.instantiate(state, signal=signal)
        return item


# Samplers
class BatchSampler(_BatchSampler, SharedMixin):
    def __init__(self, sampler, batch_size: int, drop_last: bool = False):
        self.shared_dict = Manager().dict()
        super().__init__(sampler, batch_size, drop_last=drop_last)


class ResumableDistributedSampler(DistributedSampler):  # pragma: no cover
    def __init__(self, dataset, start_idx=None, **kwargs):
        super().__init__(dataset, **kwargs)
        # Start index, allows to resume an experiment at the index it was
        self.start_idx = start_idx // self.num_replicas if start_idx is not None else 0

    def __iter__(self):
        for i, idx in enumerate(super().__iter__()):
            if i >= self.start_idx:
                yield idx
        self.start_idx = 0  # set the index back to 0 so for the next epoch


class ResumableSequentialSampler(SequentialSampler):  # pragma: no cover
    def __init__(self, dataset, start_idx=None, **kwargs):
        super().__init__(dataset, **kwargs)
        # Start index, allows to resume an experiment at the index it was
        self.start_idx = start_idx if start_idx is not None else 0

    def __iter__(self):
        for i, idx in enumerate(super().__iter__()):
            if i >= self.start_idx:
                yield idx
        self.start_idx = 0  # set the index back to 0 so for the next epoch
