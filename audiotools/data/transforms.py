from inspect import signature
from typing import List

import torch
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
        for k in self.keys:
            if k != "signal" and self.prefix is not None:
                k_with_prefix = f"{self.prefix}.{k}"
            else:
                k_with_prefix = k
            assert k_with_prefix in batch.keys(), f"{k_with_prefix} not in batch"

            batch[k] = batch[k_with_prefix]

        if "original" not in batch:
            batch["original"] = batch["signal"].clone()

        return batch

    def _transform(self, batch: dict):
        return batch

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        return {}

    def transform(self, batch: dict):
        batch = self.prepare(batch)
        mask = batch["mask"]

        if torch.any(mask):
            masked_batch = batch.copy()
            for k in self.keys + ["original"]:
                if isinstance(batch[k], AudioSignal):
                    _mask = mask
                    if len(_mask.shape) == 0:
                        _mask = _mask.unsqueeze(0)
                    masked_batch[k] = batch[k][_mask]
                elif torch.is_tensor(batch[k]):
                    masked_batch[k] = batch[k][mask]

            masked_batch = self._transform(masked_batch)

            batch["signal"][mask] = masked_batch["signal"]
            batch["original"][mask] = masked_batch["original"]

        return batch

    def __call__(self, batch: dict):
        return self.transform(batch)

    def instantiate(self, state: RandomState, signal: AudioSignal = None):
        state = util.random_state(state)

        needs_signal = "signal" in set(signature(self._instantiate).parameters.keys())
        kwargs = {}
        if needs_signal:
            kwargs = {"signal": signal}

        params = self._instantiate(state, **kwargs)

        def _prefix(k):
            if isinstance(self, META_TRANSFORMS):
                return k
            return f"{self.prefix}.{k}"

        for k in list(params.keys()):
            v = params[k]
            pk = _prefix(k)
            if isinstance(v, AudioSignal) or torch.is_tensor(v):
                params[_prefix(k)] = v
            else:
                params[_prefix(k)] = tt(v)
            if k != pk:
                params.pop(k)

        mask = state.rand() <= self.prob
        params[f"{self.prefix}.mask"] = tt(mask)

        return params


class Compose(BaseTransform):
    def __init__(self, transforms: list, prob: float = 1.0):
        super().__init__(prob=prob)
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


META_TRANSFORMS = (Compose,)


class ClippingDistortion(BaseTransform):
    def __init__(self, min: float = 0.0, max: float = 0.1, prob: float = 1.0):
        keys = ["perc"]
        super().__init__(keys=keys, prob=prob)

        self.min = min
        self.max = max

    def _instantiate(self, state: RandomState):
        return {"perc": state.uniform(self.min, self.max)}

    def _transform(self, batch):
        batch["signal"] = batch["signal"].clip_distortion(batch["perc"])
        return batch


class Equalizer(BaseTransform):
    def __init__(self, eq_amount: float = 1.0, n_bands: int = 6, prob: float = 1.0):
        keys = ["eq"]
        super().__init__(keys=keys, prob=prob)

        self.eq_amount = eq_amount
        self.n_bands = n_bands

    def _instantiate(self, state: RandomState):
        eq = -self.eq_amount * state.rand(self.n_bands)
        return {"eq": eq}

    def _transform(self, batch):
        batch["signal"] = batch["signal"].equalizer(batch["eq"])
        return batch


class Quantization(BaseTransform):
    def __init__(self, min: int = 8, max: int = 32, prob: float = 1.0):
        keys = ["channels"]
        super().__init__(keys=keys, prob=prob)

        self.min = min
        self.max = max

    def _instantiate(self, state: RandomState):
        return {"channels": state.randint(self.min, self.max)}

    def _transform(self, batch: dict):
        batch["signal"] = batch["signal"].quantization(batch["channels"])
        return batch


class MuLawQuantization(BaseTransform):
    def __init__(self, min: int = 8, max: int = 32, prob: float = 1.0):
        keys = ["channels"]
        super().__init__(keys=keys, prob=prob)

        self.min = min
        self.max = max

    def _instantiate(self, state: RandomState):
        return {"channels": state.randint(self.min, self.max)}

    def _transform(self, batch: dict):
        batch["signal"] = batch["signal"].mulaw_quantization(batch["channels"])
        return batch


class BackgroundNoise(BaseTransform):
    def __init__(
        self,
        min: float = 10.0,
        max: float = 30,
        csv_files: List[str] = None,
        eq_amount: float = 1.0,
        n_bands: int = 3,
        prob: float = 1.0,
    ):
        keys = ["eq", "snr", "bg_signal"]
        super().__init__(keys=keys, prob=prob)

        self.min = min
        self.max = max
        self.eq_amount = eq_amount
        self.n_bands = n_bands
        self.audio_files = util.read_csv(csv_files)

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        eq = -self.eq_amount * state.rand(self.n_bands)
        snr = state.uniform(self.min, self.max)

        bg_path = util.choose_from_list_of_lists(state, self.audio_files)["path"]

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
        min: float = 0.0,
        max: float = 30,
        csv_files: List[str] = None,
        eq_amount: float = 1.0,
        n_bands: int = 6,
        prob: float = 1.0,
    ):
        keys = ["eq", "drr", "ir_signal"]
        super().__init__(keys=keys, prob=prob)

        self.min = min
        self.max = max
        self.eq_amount = eq_amount
        self.n_bands = n_bands
        self.audio_files = util.read_csv(csv_files)

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        eq = -self.eq_amount * state.rand(self.n_bands)
        drr = state.uniform(self.min, self.max)

        ir_path = util.choose_from_list_of_lists(state, self.audio_files)["path"]
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
        min: float = -12,
        max: float = 0.0,
        prob: float = 1.0,
        apply_to_original: bool = True,
    ):
        keys = ["db"]
        super().__init__(keys=keys, prob=prob)

        self.min = min
        self.max = max
        self.apply_to_original = apply_to_original

    def _instantiate(self, state: RandomState):
        return {"db": state.uniform(self.min, self.max)}

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
        super().__init__(keys=keys, prob=prob)

        self.db = db
        self.apply_to_original = apply_to_original

    def _transform(self, batch):
        db_change = self.db - batch["loudness"]

        batch["signal"] = batch["signal"].volume_change(db_change)
        if self.apply_to_original:
            batch["original"] = batch["original"].volume_change(db_change)
        return batch


class Silence(BaseTransform):
    def __init__(self, prob: float = 0.1, apply_to_original: bool = True):
        super().__init__(prob=prob)

        self.apply_to_original = apply_to_original

    def _transform(self, batch: dict):
        batch["signal"] = torch.zeros_like(batch["signal"].audio_data)
        if self.apply_to_original:
            batch["original"] = torch.zeros_like(batch["original"].audio_data)
        return super()._transform(batch)


class LowPass(BaseTransform):
    def __init__(self, min: float = 0.0, max: float = 8000, prob: float = 1):
        keys = ["cutoff"]
        super().__init__(keys=keys, prob=prob)

        self.min = min
        self.max = max

    def _instantiate(self, state: RandomState):
        return {"cutoff": state.uniform(self.min, self.max)}

    def _transform(self, batch):
        batch["signal"] = batch["signal"].low_pass(batch["cutoff"])
        return batch


class HighPass(BaseTransform):
    def __init__(self, min: float = 0.0, max: float = 8000, prob: float = 1):
        keys = ["cutoff"]
        super().__init__(keys=keys, prob=prob)

        self.min = min
        self.max = max

    def _instantiate(self, state: RandomState):
        return {"cutoff": state.uniform(self.min, self.max)}

    def _transform(self, batch):
        batch["signal"] = batch["signal"].high_pass(batch["cutoff"])
        return batch
