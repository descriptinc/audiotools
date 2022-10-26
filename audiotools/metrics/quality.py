import numpy as np
import torch

from .. import AudioSignal


def stoi(
    estimates: AudioSignal,
    references: AudioSignal,
    extended: int = False,
):
    """Short term objective intelligibility
    Computes the STOI (See [1][2]) of a denoised signal compared to a clean
    signal, The output is expected to have a monotonic relation with the
    subjective speech-intelligibility, where a higher score denotes better
    speech intelligibility. Uses pystoi under the hood.

    Parameters
    ----------
    estimates : AudioSignal
        Denoised speech
    references : AudioSignal
        Clean original speech
    extended : int, optional
        Boolean, whether to use the extended STOI described in [3], by default False

    Returns
    -------
    float
        Short time objective intelligibility measure between clean and
        denoised speech

    References
    ----------
    1.  C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time
        Objective Intelligibility Measure for Time-Frequency Weighted Noisy
        Speech', ICASSP 2010, Texas, Dallas.
    2.  C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for
        Intelligibility Prediction of Time-Frequency Weighted Noisy Speech',
        IEEE Transactions on Audio, Speech, and Language Processing, 2011.
    3.  Jesper Jensen and Cees H. Taal, 'An Algorithm for Predicting the
        Intelligibility of Speech Masked by Modulated Noise Maskers',
        IEEE Transactions on Audio, Speech and Language Processing, 2016.
    """
    import pystoi

    estimates = estimates.clone().to_mono()
    references = references.clone().to_mono()

    stois = []
    for i in range(estimates.batch_size):
        _stoi = pystoi.stoi(
            references.audio_data[i, 0].detach().cpu().numpy(),
            estimates.audio_data[i, 0].detach().cpu().numpy(),
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
    """_summary_

    Parameters
    ----------
    estimates : AudioSignal
        Degraded AudioSignal
    references : AudioSignal
        Reference AudioSignal
    mode : str, optional
        'wb' (wide-band) or 'nb' (narrow-band), by default "wb"
    target_sr : float, optional
        Target sample rate, by default 16000

    Returns
    -------
    float
        PESQ score: P.862.2 Prediction (MOS-LQO)
    """
    from pesq import pesq as pesq_fn

    estimates = estimates.clone().to_mono().resample(target_sr)
    references = references.clone().to_mono().resample(target_sr)

    pesqs = []
    for i in range(estimates.batch_size):
        _pesq = pesq_fn(
            estimates.sample_rate,
            references.audio_data[i, 0].detach().cpu().numpy(),
            estimates.audio_data[i, 0].detach().cpu().numpy(),
            mode,
        )
        pesqs.append(_pesq)
    return torch.from_numpy(np.array(pesqs))
