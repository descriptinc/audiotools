from inspect import signature
from typing import List

import torch
from flatten_dict import flatten
from flatten_dict import unflatten
from numpy.random import RandomState

from ..core import AudioSignal
from ..core import util

tt = torch.tensor


class BaseTransform:
    def __init__(self, keys: list = [], prob: float = 1.0):
        self.keys = keys + ["signal", "mask"]
        self.prob = prob

        self.prefix = self.__class__.__name__

    def prepare(self, batch: dict):

        if "original" not in batch:
            batch["original"] = batch["signal"].clone()

        sub_batch = batch[self.prefix]
        sub_batch["signal"] = batch["signal"]
        sub_batch["original"] = batch["original"]
        for k in self.keys:
            assert k in sub_batch.keys(), f"{k} not in batch"

        return sub_batch

    def _transform(self, batch: dict):
        return batch

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        return {}

    def transform(self, batch: dict):
        tfm_batch = self.prepare(batch)
        mask = tfm_batch["mask"]

        def apply_mask(batch, mask):
            masked_batch = {k: v[mask] for k, v in flatten(batch).items()}
            return unflatten(masked_batch)

        if torch.any(mask):
            tfm_batch = apply_mask(tfm_batch, mask)
            tfm_batch = self._transform(tfm_batch)
            batch["signal"][mask] = tfm_batch["signal"]
            batch["original"][mask] = tfm_batch["original"]
        return batch

    def __call__(self, batch: dict):
        return self.transform(batch)

    def instantiate(self, state: RandomState, signal: AudioSignal = None):
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
        params = {self.prefix: params}

        return params


class Compose(BaseTransform):
    def __init__(self, transforms: list, prob: float = 1.0):
        keys = [tfm.prefix for tfm in transforms]
        super().__init__(keys=keys, prob=prob)
        self.transforms = transforms

    def _transform(self, batch: dict):
        for transform in self.transforms:
            batch = transform(batch)
        return batch

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        parameters = {}
        for transform in self.transforms:
            parameters.update(transform.instantiate(state, signal=signal))
        return parameters


class ClippingDistortion(BaseTransform):
    def __init__(self, perc: tuple = ("uniform", 0.0, 0.1), prob: float = 1.0):
        keys = ["perc"]
        super().__init__(keys=keys, prob=prob)

        self.perc = perc

    def _instantiate(self, state: RandomState):
        return {"perc": util.sample_from_dist(self.perc, state)}

    def _transform(self, batch):
        batch["signal"] = batch["signal"].clip_distortion(batch["perc"])
        return batch


class Equalizer(BaseTransform):
    def __init__(
        self, eq_amount: tuple = ("const", 1.0), n_bands: int = 6, prob: float = 1.0
    ):
        keys = ["eq"]
        super().__init__(keys=keys, prob=prob)

        self.eq_amount = eq_amount
        self.n_bands = n_bands

    def _instantiate(self, state: RandomState):
        eq_amount = util.sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        return {"eq": eq}

    def _transform(self, batch):
        batch["signal"] = batch["signal"].equalizer(batch["eq"])
        return batch


class Quantization(BaseTransform):
    def __init__(
        self, channels: tuple = ("choice", [8, 32, 128, 256, 1024]), prob: float = 1.0
    ):
        keys = ["channels"]
        super().__init__(keys=keys, prob=prob)

        self.channels = channels

    def _instantiate(self, state: RandomState):
        return {"channels": util.sample_from_dist(self.channels, state)}

    def _transform(self, batch: dict):
        batch["signal"] = batch["signal"].quantization(batch["channels"])
        return batch


class MuLawQuantization(BaseTransform):
    def __init__(
        self, channels: tuple = ("choice", [8, 32, 128, 256, 1024]), prob: float = 1.0
    ):
        keys = ["channels"]
        super().__init__(keys=keys, prob=prob)

        self.channels = channels

    def _instantiate(self, state: RandomState):
        return {"channels": util.sample_from_dist(self.channels, state)}

    def _transform(self, batch: dict):
        batch["signal"] = batch["signal"].mulaw_quantization(batch["channels"])
        return batch


class BackgroundNoise(BaseTransform):
    def __init__(
        self,
        snr: tuple = ("uniform", 10.0, 30.0),
        csv_files: List[str] = None,
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 3,
        prob: float = 1.0,
    ):
        """
        min and max refer to SNR.
        """
        keys = ["eq", "snr", "bg_signal"]
        super().__init__(keys=keys, prob=prob)

        self.snr = snr
        self.eq_amount = eq_amount
        self.n_bands = n_bands
        self.audio_files = util.read_csv(csv_files)

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        eq_amount = util.sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        snr = util.sample_from_dist(self.snr, state)

        bg_path = util.choose_from_list_of_lists(state, self.audio_files)["path"]

        # Get properties of input signal to use when creating
        # background signal.
        duration = signal.signal_duration
        sample_rate = signal.sample_rate
        is_mono = signal.num_channels == 1

        bg_signal = AudioSignal.excerpt(
            bg_path, duration=duration, state=state
        ).resample(sample_rate)
        if is_mono:
            bg_signal = bg_signal.to_mono()

        return {"eq": eq, "bg_signal": bg_signal, "snr": snr}

    def _transform(self, batch: dict):
        batch["signal"] = batch["signal"].mix(
            batch["bg_signal"], batch["snr"], batch["eq"]
        )
        return batch


class RoomImpulseResponse(BaseTransform):
    def __init__(
        self,
        drr: tuple = ("uniform", 0.0, 30.0),
        csv_files: List[str] = None,
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 6,
        prob: float = 1.0,
    ):
        keys = ["eq", "drr", "ir_signal"]
        super().__init__(keys=keys, prob=prob)

        self.drr = drr
        self.eq_amount = eq_amount
        self.n_bands = n_bands
        self.audio_files = util.read_csv(csv_files)

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        eq_amount = util.sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        drr = util.sample_from_dist(self.drr, state)

        ir_path = util.choose_from_list_of_lists(state, self.audio_files)["path"]

        # Get properties of input signal to use when creating
        # background signal.
        sample_rate = signal.sample_rate
        is_mono = signal.num_channels == 1

        ir_signal = (
            AudioSignal(ir_path, duration=1.0)
            .resample(sample_rate)
            .zero_pad_to(sample_rate)
        )
        if is_mono:
            ir_signal = ir_signal.to_mono()

        return {"eq": eq, "ir_signal": ir_signal, "drr": drr}

    def _transform(self, batch: dict):
        batch["signal"] = batch["signal"].apply_ir(
            batch["ir_signal"], batch["drr"], batch["eq"]
        )
        return batch


class VolumeChange(BaseTransform):
    def __init__(
        self,
        db: tuple = ("uniform", -12.0, 0.0),
        prob: float = 1.0,
        apply_to_original: bool = True,
    ):
        keys = ["db"]
        if apply_to_original:
            keys += ["original"]
        super().__init__(keys=keys, prob=prob)

        self.db = db
        self.apply_to_original = apply_to_original

    def _instantiate(self, state: RandomState):
        return {"db": util.sample_from_dist(self.db, state)}

    def _transform(self, batch):
        batch["signal"] = batch["signal"].volume_change(batch["db"])
        if self.apply_to_original:
            batch["original"] = batch["original"].volume_change(batch["db"])
        return batch


class VolumeNorm(BaseTransform):
    def __init__(
        self, db: float = -24, apply_to_original: bool = True, prob: float = 1.0
    ):
        keys = ["loudness"]
        if apply_to_original:
            keys += ["original"]
        super().__init__(keys=keys, prob=prob)

        self.db = db
        self.apply_to_original = apply_to_original

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        return {"loudness": signal.metadata["file_loudness"]}

    def _transform(self, batch):
        db_change = self.db - batch["loudness"]

        batch["signal"] = batch["signal"].volume_change(db_change)
        if self.apply_to_original:
            batch["original"] = batch["original"].volume_change(db_change)
        return batch


class Silence(BaseTransform):
    def __init__(self, prob: float = 0.1, apply_to_original: bool = True):
        keys = ["original"] if apply_to_original else []
        super().__init__(keys=keys, prob=prob)

        self.apply_to_original = apply_to_original

    def _transform(self, batch: dict):
        batch["signal"] = AudioSignal(
            torch.zeros_like(batch["signal"].audio_data),
            sample_rate=batch["signal"].sample_rate,
            stft_params=batch["signal"].stft_params,
        )
        if self.apply_to_original:
            _loudness = batch["original"]._loudness
            batch["original"] = AudioSignal(
                torch.zeros_like(batch["original"].audio_data),
                sample_rate=batch["original"].sample_rate,
                stft_params=batch["original"].stft_params,
            )
            # So that the amound of noise added is as if it wasn't silenced.
            # TODO: improve this hack
            batch["original"]._loudness = _loudness
        return super()._transform(batch)


class LowPass(BaseTransform):
    def __init__(
        self, cutoff: tuple = ("choice", [4000, 8000, 16000]), prob: float = 1
    ):
        keys = ["cutoff"]
        super().__init__(keys=keys, prob=prob)

        self.cutoff = cutoff

    def _instantiate(self, state: RandomState):
        return {"cutoff": util.sample_from_dist(self.cutoff, state)}

    def _transform(self, batch):
        batch["signal"] = batch["signal"].low_pass(batch["cutoff"])
        return batch


class HighPass(BaseTransform):
    def __init__(
        self, cutoff: tuple = ("choice", [50, 100, 250, 500, 1000]), prob: float = 1
    ):
        keys = ["cutoff"]
        super().__init__(keys=keys, prob=prob)

        self.cutoff = cutoff

    def _instantiate(self, state: RandomState):
        return {"cutoff": util.sample_from_dist(self.cutoff, state)}

    def _transform(self, batch):
        batch["signal"] = batch["signal"].high_pass(batch["cutoff"])
        return batch


class RescaleAudio(BaseTransform):
    def __init__(self, val: float = 1.0, prob: float = 1):
        super().__init__(prob=prob)

        self.val = val

    def _transform(self, batch: dict):
        batch["signal"] = batch["signal"].ensure_max_of_audio(self.val)
        return batch
