from inspect import signature
from typing import List

import torch
from numpy.random import RandomState

from ..core import AudioSignal
from ..core import util

tt = torch.tensor


class BaseTransform:
    def __init__(self, keys: list = [], prob: float = 1.0):
        self.keys = keys + ["signal"]
        self.prob = prob

    def validate(self, batch: dict):
        for k in self.keys:
            assert k in batch.keys(), f"{k} not in batch"

        if "original" not in batch:
            batch["original"] = batch["signal"].clone()

        return batch

    def get_mask(self, batch):
        mask_key = f"{self.__class__.__name__}.mask"
        mask = batch[mask_key]
        return mask

    def _transform(self, batch: dict):
        return batch

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        return {}

    def transform(self, batch: dict):
        batch = self.validate(batch)
        mask = self.get_mask(batch)

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
        mask = state.rand() <= self.prob
        params.update({f"{self.__class__.__name__}.mask": mask})

        for k, v in params.items():
            if not torch.is_tensor(v) and not isinstance(v, AudioSignal):
                params[k] = tt(v)

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


class ClippingDistortion(BaseTransform):
    def __init__(self, min: float = 0.0, max: float = 0.1, prob: float = 1.0):
        keys = ["clip_percentile"]
        super().__init__(keys=keys, prob=prob)

        self.min = min
        self.max = max

    def _instantiate(self, state: RandomState):
        return {"clip_percentile": state.uniform(self.min, self.max)}

    def _transform(self, batch):
        signal = batch["signal"]
        clip_percentile = batch["clip_percentile"]
        batch["signal"] = signal.clip_distortion(clip_percentile)
        return batch


class Equalizer(BaseTransform):
    def __init__(self, eq_amount: float = 1.0, n_bands: int = 6, prob: float = 1.0):
        keys = ["eq_curve"]
        super().__init__(keys=keys, prob=prob)

        self.eq_amount = eq_amount
        self.n_bands = n_bands

    def _instantiate(self, state: RandomState):
        eq_curve = -self.eq_amount * state.rand(self.n_bands)
        return {"eq_curve": eq_curve}

    def _transform(self, batch):
        signal = batch["signal"]
        eq_curve = batch["eq_curve"]
        batch["signal"] = signal.equalizer(eq_curve)
        return batch


class Quantization(BaseTransform):
    def __init__(self, min: int = 8, max: int = 32, prob: float = 1.0):
        keys = ["quantization_channels"]
        super().__init__(keys=keys, prob=prob)

        self.min = min
        self.max = max

    def _instantiate(self, state: RandomState):
        return {"quantization_channels": state.randint(self.min, self.max)}

    def _transform(self, batch: dict):
        signal = batch["signal"]
        quant_ch = batch["quantization_channels"]
        batch["signal"] = signal.quantization(quant_ch)
        return batch


class MuLawQuantization(BaseTransform):
    def __init__(self, min: int = 8, max: int = 32, prob: float = 1.0):
        keys = ["quantization_channels"]
        super().__init__(keys=keys, prob=prob)

        self.min = min
        self.max = max

    def _instantiate(self, state: RandomState):
        return {"quantization_channels": state.randint(self.min, self.max)}

    def _transform(self, batch: dict):
        signal = batch["signal"]
        quant_ch = batch["quantization_channels"]
        batch["signal"] = signal.mulaw_quantization(quant_ch)
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
        keys = ["bg_eq_curve", "snr", "bg_signal"]
        super().__init__(keys=keys, prob=prob)

        self.min = min
        self.max = max
        self.eq_amount = eq_amount
        self.n_bands = n_bands
        self.audio_files = util.read_csv(csv_files)

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        bg_eq_curve = -self.eq_amount * state.rand(self.n_bands)
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

        return {"bg_eq_curve": bg_eq_curve, "bg_signal": bg_signal, "snr": snr}

    def _transform(self, batch: dict):
        batch["signal"] = batch["signal"].mix(
            batch["bg_signal"], batch["snr"], batch["bg_eq_curve"]
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
        keys = ["ir_eq_curve", "drr", "ir_signal"]
        super().__init__(keys=keys, prob=prob)

        self.min = min
        self.max = max
        self.eq_amount = eq_amount
        self.n_bands = n_bands
        self.audio_files = util.read_csv(csv_files)

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        ir_eq_curve = -self.eq_amount * state.rand(self.n_bands)
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

        return {"ir_eq_curve": ir_eq_curve, "ir_signal": ir_signal, "drr": drr}

    def _transform(self, batch: dict):
        batch["signal"] = batch["signal"].apply_ir(
            batch["ir_signal"], batch["drr"], batch["ir_eq_curve"]
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
        keys = ["db_change"]
        super().__init__(keys=keys, prob=prob)

        self.min = min
        self.max = max
        self.apply_to_original = apply_to_original

    def _instantiate(self, state: RandomState):
        return {"db_change": state.uniform(self.min, self.max)}

    def _transform(self, batch):
        batch["signal"] = batch["signal"].volume_change(batch["db_change"])
        if self.apply_to_original:
            batch["original"] = batch["original"].volume_change(batch["db_change"])
        return batch


class FileLevelVolumeNorm(BaseTransform):
    def __init__(
        self, db: float = -24, apply_to_original: bool = True, prob: float = 1.0
    ):
        keys = ["file_loudness"]
        super().__init__(keys=keys, prob=prob)

        self.db = db
        self.apply_to_original = apply_to_original

    def _transform(self, batch):
        db_change = self.db - batch["file_loudness"]

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
        keys = ["lp_cutoff"]
        super().__init__(keys=keys, prob=prob)

        self.min = min
        self.max = max

    def _instantiate(self, state: RandomState):
        return {"lp_cutoff": state.uniform(self.min, self.max)}

    def _transform(self, batch):
        batch["signal"] = batch["signal"].low_pass(batch["lp_cutoff"])
        return batch


class HighPass(BaseTransform):
    def __init__(self, min: float = 0.0, max: float = 8000, prob: float = 1):
        keys = ["hp_cutoff"]
        super().__init__(keys=keys, prob=prob)

        self.min = min
        self.max = max

    def _instantiate(self, state: RandomState):
        return {"hp_cutoff": state.uniform(self.min, self.max)}

    def _transform(self, batch):
        batch["signal"] = batch["signal"].high_pass(batch["hp_cutoff"])
        return batch
