import copy
from contextlib import contextmanager
from inspect import signature
from typing import List

import numpy as np
import torch
from flatten_dict import flatten
from flatten_dict import unflatten
from numpy.random import RandomState
from yaml import load

from ..core import AudioSignal
from ..core import util
from .datasets import AudioLoader
from .. import ml

tt = torch.tensor


class BaseTransform:
    def __init__(self, keys: list = [], name: str = None, prob: float = 1.0):
        # Get keys from the _transform signature.
        tfm_keys = list(signature(self._transform).parameters.keys())

        # Filter out signal and kwargs keys.
        ignore_keys = ["signal", "kwargs"]
        tfm_keys = [k for k in tfm_keys if k not in ignore_keys]

        # Combine keys specified by the child class, the keys found in
        # _transform signature, and the mask key.
        self.keys = keys + tfm_keys + ["mask"]

        self.prob = prob

        if name is None:
            name = self.__class__.__name__
        self.name = name

    def prepare(self, batch: dict):
        sub_batch = batch[self.name]

        for k in self.keys:
            assert k in sub_batch.keys(), f"{k} not in batch"

        return sub_batch

    def _transform(self, signal):
        return signal

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        return {}

    @staticmethod
    def apply_mask(batch, mask):
        masked_batch = {k: v[mask] for k, v in flatten(batch).items()}
        return unflatten(masked_batch)

    def transform(self, signal, **kwargs):
        tfm_kwargs = self.prepare(kwargs)
        mask = tfm_kwargs["mask"]

        if torch.any(mask):
            tfm_kwargs = self.apply_mask(tfm_kwargs, mask)
            tfm_kwargs = {k: v for k, v in tfm_kwargs.items() if k != "mask"}
            signal[mask] = self._transform(signal[mask], **tfm_kwargs)

        return signal

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def instantiate(
        self,
        state: RandomState = None,
        signal: AudioSignal = None,
    ):
        state = util.random_state(state)

        # Not all instantiates need the signal. Check if signal
        # is needed before passing it in, so that the end-user
        # doesn't need to have variables they're not using flowing
        # into their function.
        needs_signal = "signal" in set(signature(self._instantiate).parameters.keys())
        kwargs = {}
        if needs_signal:
            kwargs = {"signal": signal}

        # Instantiate the parameters for the transform.
        params = self._instantiate(state, **kwargs)
        for k in list(params.keys()):
            v = params[k]
            if isinstance(v, (AudioSignal, torch.Tensor, dict)):
                params[k] = v
            else:
                params[k] = tt(v)
        mask = state.rand() <= self.prob
        params[f"mask"] = tt(mask)

        # Put the params into a nested dictionary that will be
        # used later when calling the transform. This is to avoid
        # collisions in the dictionary.
        params = {self.name: params}

        return params

    def batch_instantiate(
        self,
        states: list = None,
        signal: AudioSignal = None,
    ):
        kwargs = []
        for state in states:
            kwargs.append(self.instantiate(state, signal))
        kwargs = util.collate(kwargs)
        return kwargs


class Identity(BaseTransform):
    pass


class SpectralTransform(BaseTransform):
    def transform(self, signal, **kwargs):
        signal.stft()
        super().transform(signal, **kwargs)
        signal.istft()
        return signal


class Compose(BaseTransform):
    def __init__(self, *transforms: list, name: str = None, prob: float = 1.0):
        if isinstance(transforms[0], list):
            transforms = transforms[0]

        for i, tfm in enumerate(transforms):
            tfm.name = f"{i}.{tfm.name}"

        keys = [tfm.name for tfm in transforms]
        super().__init__(keys=keys, name=name, prob=prob)

        self.transforms = transforms
        self.transforms_to_apply = keys

    @contextmanager
    def filter(self, *names):
        old_transforms = self.transforms_to_apply
        self.transforms_to_apply = names
        yield
        self.transforms_to_apply = old_transforms

    def _transform(self, signal, **kwargs):
        for transform in self.transforms:
            if any([x in transform.name for x in self.transforms_to_apply]):
                signal = transform(signal, **kwargs)
        return signal

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        parameters = {}
        for transform in self.transforms:
            parameters.update(transform.instantiate(state, signal=signal))
        return parameters

    def __getitem__(self, idx):
        return self.transforms[idx]

    def __len__(self):
        return len(self.transforms)

    def __iter__(self):
        for transform in self.transforms:
            yield transform


class Choose(Compose):
    # Class logic is the same as Compose, but instead of applying all
    # the transforms in sequence, it applies just a single transform,
    # which is chosen for each item in the batch.
    def __init__(
        self,
        *transforms: list,
        weights: list = None,
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(*transforms, name=name, prob=prob)

        if weights is None:
            _len = len(self.transforms)
            weights = [1 / _len for _ in range(_len)]
        self.weights = np.array(weights)

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        kwargs = super()._instantiate(state, signal)
        tfm_idx = list(range(len(self.transforms)))
        tfm_idx = state.choice(tfm_idx, p=self.weights)
        one_hot = []
        for i, t in enumerate(self.transforms):
            mask = kwargs[t.name]["mask"]
            if mask.item():
                kwargs[t.name]["mask"] = tt(i == tfm_idx)
            one_hot.append(kwargs[t.name]["mask"])
        kwargs["one_hot"] = one_hot
        return kwargs


class Repeat(Compose):
    """Repeatedly applies a given transform."""

    def __init__(
        self,
        transform,
        n_repeat: int = 1,
        name: str = None,
        prob: float = 1.0,
    ):
        transforms = [copy.copy(transform) for _ in range(n_repeat)]
        super().__init__(transforms, name=name, prob=prob)

        self.n_repeat = n_repeat


class RepeatUpTo(Choose):
    """Repeats up to a given number of times."""

    def __init__(
        self,
        transform,
        max_repeat: int = 5,
        weights: list = None,
        name: str = None,
        prob: float = 1.0,
    ):
        transforms = []
        for n in range(1, max_repeat):
            transforms.append(Repeat(transform, n_repeat=n))
        super().__init__(transforms, name=name, prob=prob, weights=weights)

        self.max_repeat = max_repeat


class ClippingDistortion(BaseTransform):
    def __init__(
        self,
        perc: tuple = ("uniform", 0.0, 0.1),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.perc = perc

    def _instantiate(self, state: RandomState):
        return {"perc": util.sample_from_dist(self.perc, state)}

    def _transform(self, signal, perc):
        return signal.clip_distortion(perc)


class Equalizer(BaseTransform):
    def __init__(
        self,
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 6,
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.eq_amount = eq_amount
        self.n_bands = n_bands

    def _instantiate(self, state: RandomState):
        eq_amount = util.sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        return {"eq": eq}

    def _transform(self, signal, eq):
        return signal.equalizer(eq)


class Quantization(BaseTransform):
    def __init__(
        self,
        channels: tuple = ("choice", [8, 32, 128, 256, 1024]),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.channels = channels

    def _instantiate(self, state: RandomState):
        return {"channels": util.sample_from_dist(self.channels, state)}

    def _transform(self, signal, channels):
        return signal.quantization(channels)


class MuLawQuantization(BaseTransform):
    def __init__(
        self,
        channels: tuple = ("choice", [8, 32, 128, 256, 1024]),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.channels = channels

    def _instantiate(self, state: RandomState):
        return {"channels": util.sample_from_dist(self.channels, state)}

    def _transform(self, signal, channels):
        return signal.mulaw_quantization(channels)


class NoiseFloor(BaseTransform):
    def __init__(
        self,
        db: tuple = ("const", -50.0),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.db = db

    def _instantiate(self, state: RandomState, signal: AudioSignal):
        db = util.sample_from_dist(self.db, state)
        audio_data = state.randn(signal.num_channels, signal.signal_length)
        nz_signal = AudioSignal(audio_data, signal.sample_rate)
        nz_signal.normalize(db)
        return {"nz_signal": nz_signal}

    def _transform(self, signal, nz_signal):
        # Clone bg_signal so that transform can be repeatedly applied
        # to different signals with the same effect.
        return signal + nz_signal


class BackgroundNoise(BaseTransform):
    def __init__(
        self,
        snr: tuple = ("uniform", 10.0, 30.0),
        csv_files: List[str] = None,
        csv_weights: List[float] = None,
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 3,
        name: str = None,
        prob: float = 1.0,
        loudness_cutoff: float = None,
    ):
        super().__init__(name=name, prob=prob)

        self.snr = snr
        self.eq_amount = eq_amount
        self.n_bands = n_bands
        self.loader = AudioLoader(csv_files, csv_weights)
        self.loudness_cutoff = loudness_cutoff

    def _instantiate(self, state: RandomState, signal: AudioSignal):
        eq_amount = util.sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        snr = util.sample_from_dist(self.snr, state)

        bg_signal, _ = self.loader(
            state,
            signal.sample_rate,
            duration=signal.signal_duration,
            loudness_cutoff=self.loudness_cutoff,
            num_channels=signal.num_channels,
        )

        return {"eq": eq, "bg_signal": bg_signal, "snr": snr}

    def _transform(self, signal, bg_signal, snr, eq):
        # Clone bg_signal so that transform can be repeatedly applied
        # to different signals with the same effect.
        return signal.mix(bg_signal.clone(), snr, eq)


class CrossTalk(BaseTransform):
    def __init__(
        self,
        snr: tuple = ("uniform", 0.0, 10.0),
        csv_files: List[str] = None,
        csv_weights: List[float] = None,
        name: str = None,
        prob: float = 1.0,
        loudness_cutoff: float = -40,
    ):
        super().__init__(name=name, prob=prob)

        self.snr = snr
        self.loader = AudioLoader(csv_files, csv_weights)
        self.loudness_cutoff = loudness_cutoff

    def _instantiate(self, state: RandomState, signal: AudioSignal):
        snr = util.sample_from_dist(self.snr, state)
        crosstalk_signal, _ = self.loader(
            state,
            signal.sample_rate,
            duration=signal.signal_duration,
            loudness_cutoff=self.loudness_cutoff,
            num_channels=signal.num_channels,
        )

        return {"crosstalk_signal": crosstalk_signal, "snr": snr}

    def _transform(self, signal, crosstalk_signal, snr):
        # Clone bg_signal so that transform can be repeatedly applied
        # to different signals with the same effect.
        loudness = signal.loudness()
        mix = signal.mix(crosstalk_signal.clone(), snr)
        mix.normalize(loudness)
        return mix


class RoomImpulseResponse(BaseTransform):
    def __init__(
        self,
        drr: tuple = ("uniform", 0.0, 30.0),
        csv_files: List[str] = None,
        csv_weights: List[float] = None,
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 6,
        name: str = None,
        prob: float = 1.0,
        use_original_phase: bool = False,
        offset: float = 0.0,
        duration: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.drr = drr
        self.eq_amount = eq_amount
        self.n_bands = n_bands
        self.use_original_phase = use_original_phase

        self.loader = AudioLoader(csv_files, csv_weights)
        self.offset = offset
        self.duration = duration

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        eq_amount = util.sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        drr = util.sample_from_dist(self.drr, state)

        ir_signal, _ = self.loader(
            state,
            signal.sample_rate,
            offset=self.offset,
            duration=self.duration,
            loudness_cutoff=None,
            num_channels=signal.num_channels,
        )
        ir_signal.zero_pad_to(signal.sample_rate)

        return {"eq": eq, "ir_signal": ir_signal, "drr": drr}

    def _transform(self, signal, ir_signal, drr, eq):
        # Clone ir_signal so that transform can be repeatedly applied
        # to different signals with the same effect.
        return signal.apply_ir(
            ir_signal.clone(), drr, eq, use_original_phase=self.use_original_phase
        )


class VolumeChange(BaseTransform):
    def __init__(
        self,
        db: tuple = ("uniform", -12.0, 0.0),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)
        self.db = db

    def _instantiate(self, state: RandomState):
        return {"db": util.sample_from_dist(self.db, state)}

    def _transform(self, signal, db):
        return signal.volume_change(db)


class VolumeNorm(BaseTransform):
    def __init__(
        self,
        db: tuple = ("const", -24),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.db = db

    def _instantiate(self, state: RandomState):
        return {"db": util.sample_from_dist(self.db, state)}

    def _transform(self, signal, db):
        return signal.normalize(db)


class GlobalVolumeNorm(BaseTransform):
    def __init__(
        self,
        db: tuple = ("const", -24),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.db = db

    def _instantiate(self, state: RandomState, signal: AudioSignal):
        assert "loudness" in signal.metadata
        db = util.sample_from_dist(self.db, state)
        db_change = db - float(signal.metadata["loudness"])
        return {"db": db_change}

    def _transform(self, signal, db):
        return signal.volume_change(db)


class Silence(BaseTransform):
    def __init__(self, name: str = None, prob: float = 0.1):
        super().__init__(name=name, prob=prob)

    def _transform(self, signal):
        _loudness = signal._loudness
        signal = AudioSignal(
            torch.zeros_like(signal.audio_data),
            sample_rate=signal.sample_rate,
            stft_params=signal.stft_params,
        )
        # So that the amound of noise added is as if it wasn't silenced.
        # TODO: improve this hack
        signal._loudness = _loudness

        return signal


class LowPass(BaseTransform):
    def __init__(
        self,
        cutoff: tuple = ("choice", [4000, 8000, 16000]),
        zeros: int = 51,
        name: str = None,
        prob: float = 1,
    ):
        """Applies a LowPass filter.

        Parameters
        ----------
        cutoff : tuple, optional
            Cutoff frequency distribution,
            by default ("choice", [4000, 8000, 16000])
        zeros : int, optional
            Number of zero-crossings in filter, argument to
            julius.LowPassFilters, by default 51
        name : str, optional
            Name of the transform, by default None
        prob : float, optional
            Probability transform is applied, by default 1
        """
        super().__init__(name=name, prob=prob)

        self.cutoff = cutoff
        self.zeros = zeros

    def _instantiate(self, state: RandomState):
        return {"cutoff": util.sample_from_dist(self.cutoff, state)}

    def _transform(self, signal, cutoff):
        return signal.low_pass(cutoff, zeros=self.zeros)


class HighPass(BaseTransform):
    def __init__(
        self,
        cutoff: tuple = ("choice", [50, 100, 250, 500, 1000]),
        zeros: int = 51,
        name: str = None,
        prob: float = 1,
    ):
        """Applies a HighPass filter.

        Parameters
        ----------
        cutoff : tuple, optional
            Cutoff frequency distribution,
            by default ("choice", [50, 100, 250, 500, 1000])
        zeros : int, optional
            Number of zero-crossings in filter, argument to
            julius.LowPassFilters, by default 51
        name : str, optional
            Name of the transform, by default None
        prob : float, optional
            Probability transform is applied, by default 1
        """
        super().__init__(name=name, prob=prob)

        self.cutoff = cutoff
        self.zeros = zeros

    def _instantiate(self, state: RandomState):
        return {"cutoff": util.sample_from_dist(self.cutoff, state)}

    def _transform(self, signal, cutoff):
        return signal.high_pass(cutoff, zeros=self.zeros)


class RescaleAudio(BaseTransform):
    def __init__(self, val: float = 1.0, name: str = None, prob: float = 1):
        super().__init__(name=name, prob=prob)

        self.val = val

    def _transform(self, signal):
        return signal.ensure_max_of_audio(self.val)


class ShiftPhase(SpectralTransform):
    def __init__(
        self,
        shift: tuple = ("uniform", -np.pi, np.pi),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(name=name, prob=prob)
        self.shift = shift

    def _instantiate(self, state: RandomState):
        return {"shift": util.sample_from_dist(self.shift, state)}

    def _transform(self, signal, shift):
        return signal.shift_phase(shift)


class InvertPhase(ShiftPhase):
    def __init__(self, name: str = None, prob: float = 1):
        super().__init__(shift=("const", np.pi), name=name, prob=prob)


class CorruptPhase(SpectralTransform):
    def __init__(
        self, scale: tuple = ("uniform", 0, np.pi), name: str = None, prob: float = 1
    ):
        super().__init__(name=name, prob=prob)
        self.scale = scale

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        scale = util.sample_from_dist(self.scale, state)
        corruption = state.normal(scale=scale, size=signal.phase.shape[1:])
        return {"corruption": corruption.astype("float32")}

    def _transform(self, signal, corruption):
        return signal.shift_phase(shift=corruption)


class FrequencyMask(SpectralTransform):
    def __init__(
        self,
        f_center: tuple = ("uniform", 0.0, 1.0),
        f_width: tuple = ("const", 0.1),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(name=name, prob=prob)
        self.f_center = f_center
        self.f_width = f_width

    def _instantiate(self, state: RandomState, signal: AudioSignal):
        f_center = util.sample_from_dist(self.f_center, state)
        f_width = util.sample_from_dist(self.f_width, state)

        fmin = max(f_center - (f_width / 2), 0.0)
        fmax = min(f_center + (f_width / 2), 1.0)

        fmin_hz = (signal.sample_rate / 2) * fmin
        fmax_hz = (signal.sample_rate / 2) * fmax

        return {"fmin_hz": fmin_hz, "fmax_hz": fmax_hz}

    def _transform(self, signal, fmin_hz: float, fmax_hz: float):
        return signal.mask_frequencies(fmin_hz=fmin_hz, fmax_hz=fmax_hz)


class TimeMask(SpectralTransform):
    def __init__(
        self,
        t_center: tuple = ("uniform", 0.0, 1.0),
        t_width: tuple = ("const", 0.025),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(name=name, prob=prob)
        self.t_center = t_center
        self.t_width = t_width

    def _instantiate(self, state: RandomState, signal: AudioSignal):
        t_center = util.sample_from_dist(self.t_center, state)
        t_width = util.sample_from_dist(self.t_width, state)

        tmin = max(t_center - (t_width / 2), 0.0)
        tmax = min(t_center + (t_width / 2), 1.0)

        tmin_s = signal.signal_duration * tmin
        tmax_s = signal.signal_duration * tmax
        return {"tmin_s": tmin_s, "tmax_s": tmax_s}

    def _transform(self, signal, tmin_s: float, tmax_s: float):
        return signal.mask_timesteps(tmin_s=tmin_s, tmax_s=tmax_s)


class MaskLowMagnitudes(SpectralTransform):
    def __init__(
        self,
        db_cutoff: tuple = ("uniform", -10, 10),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(name=name, prob=prob)
        self.db_cutoff = db_cutoff

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        return {"db_cutoff": util.sample_from_dist(self.db_cutoff, state)}

    def _transform(self, signal, db_cutoff: float):
        return signal.mask_low_magnitudes(db_cutoff)


class Smoothing(BaseTransform):
    def __init__(
        self,
        window_type: tuple = ("const", "average"),
        window_length: tuple = ("choice", [8, 16, 32, 64, 128, 256, 512]),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(name=name, prob=prob)
        self.window_type = window_type
        self.window_length = window_length

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        window_type = util.sample_from_dist(self.window_type, state)
        window_length = util.sample_from_dist(self.window_length, state)
        window = signal.get_window(
            window_type=window_type, window_length=window_length, device="cpu"
        )
        return {"window": AudioSignal(window, signal.sample_rate)}

    def _transform(self, signal, window):
        sscale = signal.audio_data.abs().max(dim=-1, keepdim=True).values
        sscale[sscale == 0.0] = 1.0

        out = signal.convolve(window)

        oscale = out.audio_data.abs().max(dim=-1, keepdim=True).values
        oscale[oscale == 0.0] = 1.0

        out = out * (sscale / oscale)
        return out


class TimeNoise(TimeMask):
    def __init__(
        self,
        t_center: tuple = ("uniform", 0.0, 1.0),
        t_width: tuple = ("const", 0.025),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(t_center=t_center, t_width=t_width, name=name, prob=prob)

    def _transform(self, signal, tmin_s: float, tmax_s: float):
        signal = signal.mask_timesteps(tmin_s=tmin_s, tmax_s=tmax_s, val=0.0)
        mag, phase = signal.magnitude, signal.phase

        mag_r, phase_r = torch.randn_like(mag), torch.randn_like(phase)
        mask = (mag == 0.0) * (phase == 0.0)

        mag[mask] = mag_r[mask]
        phase[mask] = phase_r[mask]

        signal.magnitude = mag
        signal.phase = phase
        return signal


class FrequencyNoise(FrequencyMask):
    def __init__(
        self,
        f_center: tuple = ("uniform", 0.0, 1.0),
        f_width: tuple = ("const", 0.1),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(f_center=f_center, f_width=f_width, name=name, prob=prob)

    def _transform(self, signal, fmin_hz: float, fmax_hz: float):
        signal = signal.mask_frequencies(fmin_hz=fmin_hz, fmax_hz=fmax_hz)
        mag, phase = signal.magnitude, signal.phase

        mag_r, phase_r = torch.randn_like(mag), torch.randn_like(phase)
        mask = (mag == 0.0) * (phase == 0.0)

        mag[mask] = mag_r[mask]
        phase[mask] = phase_r[mask]

        signal.magnitude = mag
        signal.phase = phase
        return signal

class SpectralDenoising(Equalizer):
    def __init__(
        self,
        eq_amount: tuple = ("const", 1.0),
        denoise_amount: tuple = ("uniform", 0.8, 1.0),
        nz_volume: float = -40,
        n_bands: int = 6,
        n_freq: int = 3, 
        n_time: int = 5,
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(
            eq_amount=eq_amount, 
            n_bands=n_bands,
            name=name, 
            prob=prob
        )

        self.nz_volume = nz_volume
        self.denoise_amount = denoise_amount
        self.spectral_gate = ml.layers.SpectralGate(n_freq, n_time)

    def _transform(self, signal, nz, eq, denoise_amount):
        nz = nz.normalize(self.nz_volume).equalizer(eq)
        self.spectral_gate = self.spectral_gate.to(signal.device)
        signal = self.spectral_gate(signal, nz, denoise_amount)
        return signal

    def _instantiate(self, state: RandomState):
        kwargs = super()._instantiate(state)
        kwargs["denoise_amount"] = util.sample_from_dist(
            self.denoise_amount, state
        )
        kwargs["nz"] = AudioSignal(state.randn(22050), 44100)
        return kwargs
