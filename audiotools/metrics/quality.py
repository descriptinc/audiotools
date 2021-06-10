import numpy as np
import pystoi
import torch

from .. import AudioSignal


def stoi(
    estimates: AudioSignal,
    references: AudioSignal,
    extended: int = False,
):
    estimates = estimates.deepcopy().to_mono()
    references = references.deepcopy().to_mono()

    stois = []
    for i in range(estimates.batch_size):
        _stoi = pystoi.stoi(
            references[i, 0].detach().cpu().numpy(),
            estimates[i, 0].detach().cpu().numpy(),
            references.sample_rate,
            extended=extended,
        )
        stois.append(_stoi)
    return torch.from_numpy(np.array(stois))


def pesq(
    estimates: AudioSignal,
    references: AudioSignal,
    mode: str = "wb",
    target_sr: float = 16000,
):
    from pesq import pesq as pesq_fn

    estimates = estimates.deepcopy().to_mono().resample(target_sr)
    references = references.deepcopy().to_mono().resample(target_sr)

    pesqs = []
    for i in range(estimates.batch_size):
        _pesq = pesq_fn(
            estimates.sample_rate,
            references[i, 0].detach().cpu().numpy(),
            estimates[i, 0].detach().cpu().numpy(),
            mode,
        )
        pesqs.append(_pesq)
    return torch.from_numpy(np.array(pesqs))
