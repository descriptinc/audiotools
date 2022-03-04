import torch

from ..core import AudioSignal


def collate(list_of_dicts):
    dict_of_lists = {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}
    batch = {}
    for k, v in dict_of_lists.items():
        if isinstance(v, list):
            if all(isinstance(s, AudioSignal) for s in v):
                batch[k] = AudioSignal.batch(v, pad_signals=True)
                batch[k].loudness()
            else:
                # Borrow the default collate fn from torch.
                batch[k] = torch.utils.data._utils.collate.default_collate(v)
    return batch
