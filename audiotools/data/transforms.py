import math
from inspect import signature
from readline import parse_and_bind
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
        if mask_key not in batch:
            return slice(None, None, None)
        mask = batch[mask_key]
        return mask

    def _transform(self, batch: dict):
        raise NotImplementedError

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        raise NotImplementedError

    def transform(self, batch: dict):
        batch = self.validate(batch)
        mask = self.get_mask(batch)

        if torch.any(mask):
            masked_batch = batch.copy()
            for k in self.keys:
                if isinstance(batch[k], AudioSignal):
                    _mask = mask
                    if len(_mask.shape) == 0:
                        _mask = _mask.unsqueeze(0)
                    masked_batch[k] = batch[k][_mask]
                elif torch.is_tensor(batch[k]):
                    masked_batch[k] = batch[k][mask]

            masked_batch = self._transform(masked_batch)
            batch["signal"][mask] = masked_batch["signal"]

        return batch

    def __call__(self, batch: dict):
        return self.transform(batch)

    def instantiate(self, state: RandomState, signal: AudioSignal = None):
        state = util.random_state(state)
        params = self._instantiate(state, signal)
        mask = state.rand() <= self.prob
        params.update({f"{self.__class__.__name__}.mask": mask})

        for k, v in params.items():
            if not torch.is_tensor(v):
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

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        return {"clip_percentile": state.uniform(self.min, self.max)}

    def _transform(self, batch):
        signal = batch["signal"]
        clip_percentile = batch["clip_percentile"]
        batch["signal"] = signal.clip_distortion(clip_percentile)
        return batch


class Equalizer(BaseTransform):
    def __init__(self, eq_amount: float = 1.0, n_bands: int = 6, prob: float = 1.0):
        super().__init__(prob=prob)

        self.eq_amount = eq_amount
        self.n_bands = n_bands

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        eq_curve = -self.eq_amount * state.rand(self.n_bands)
        return {"eq_curve": eq_curve}

    def _transform(self, batch):
        signal = batch["signal"]
        eq_curve = batch["eq_curve"]
        batch["signal"] = signal.equalizer(eq_curve)
        return batch


class Quantization(BaseTransform):
    def __init__(self, min: int = 8, max: int = 32, prob: float = 1.0):
        super().__init__(prob=prob)

        self.min = min
        self.max = max

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        return {"quantization_channels": state.randint(self.min, self.max)}

    def _transform(self, batch: dict):
        signal = batch["signal"]
        quant_ch = batch["quantization_channels"]
        batch["signal"] = signal.quantization(quant_ch)
        return batch


class MuLawQuantization(BaseTransform):
    def __init__(self, min: int = 8, max: int = 32, prob: float = 1.0):
        super().__init__(prob=prob)

        self.min = min
        self.max = max

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        return {"quantization_channels": state.randint(self.min, self.max)}

    def _transform(self, batch: dict):
        signal = batch["signal"]
        quant_ch = batch["quantization_channels"]
        batch["signal"] = signal.quantization(quant_ch)
        return batch
