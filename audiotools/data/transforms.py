from collections import defaultdict
from inspect import signature
from typing import List

import numpy as np
import torch
from flatten_dict import flatten
from flatten_dict import unflatten
from numpy.random import RandomState

from ..core import AudioSignal
from ..core import util

tt = torch.tensor


class BaseTransform:
    def __init__(self, keys: list = [], signal_keys: list = None, prob: float = 1.0):
        self.keys = keys + ["mask"]
        self.signal_keys = ["signal"] if signal_keys is None else signal_keys
        self.prob = prob

        self.prefix = self.__class__.__name__

    def prepare(self, batch: dict):
        if "original" not in batch:
            batch["original"] = batch["signal"].clone()

        sub_batch = batch[self.prefix]
        # Signals to apply transformation to.
        sub_batch["signals"] = {}

        for k in self.signal_keys:
            assert k in batch, f"{k} (AudioSignal) not in batch"
            sub_batch["signals"][k] = batch[k]

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

    def transform(self, batch: dict):
        tfm_batch = self.prepare(batch)
        mask = tfm_batch["mask"]

        if torch.any(mask):
            signals = tfm_batch.pop("signals")
            tfm_batch = self.apply_mask(tfm_batch, mask)
            kwargs = {k: v for k, v in tfm_batch.items() if k != "mask"}
            for k, signal in signals.items():
                kwargs["signal"] = signal[mask]
                batch[k][mask] = self._transform(**kwargs)

        return batch

    def __call__(self, batch: dict):
        return self.transform(batch)

    def instantiate(
        self, state: RandomState, signal: AudioSignal = None, n_params: int = 1
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

        all_params = []
        for _ in range(n_params):
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

            all_params.append(params)

        if n_params > 1:
            all_params = util.collate(all_params)
        else:
            all_params = all_params[0]

        # Put the params into a nested dictionary that will be
        # used later when calling the transform. This is to avoid
        # collisions in the dictionary.
        params = {self.prefix: all_params}

        return params


class Compose(BaseTransform):
    def __init__(self, transforms: list, prob: float = 1.0):
        keys = []
        signal_keys = []
        tfm_counts = defaultdict(lambda: 0)
        for tfm in transforms:
            prefix = tfm.prefix
            tfm_counts[prefix] += 1
            prefix = f"{prefix}.{tfm_counts[prefix]}"

            tfm.prefix = prefix
            keys.append(prefix)
            signal_keys = signal_keys + tfm.signal_keys

        signal_keys = list(set(signal_keys))

        super().__init__(keys=keys, signal_keys=signal_keys, prob=prob)
        self.transforms = transforms

    def transform(self, batch: dict):
        """This is a specific transform for Compose, which must pass
        the batch through to each of its transforms as a dictionary,
        to avoid double work. While other transforms pass the specific
        arguments to the _transform function (like signal=signal, arg=arg),
        Compose doesn't actually do anything by itself, so we pass the batch
        as a dictionary to the underlying transforms.

        Parameters
        ----------
        batch : dict
            Batch containing signals and transform args.

        Returns
        -------
        dict
            Output dictionary with transformed signals.
        """
        tfm_batch = self.prepare(batch)
        signals = tfm_batch.pop("signals")
        tfm_batch.update(signals)
        mask = tfm_batch["mask"]

        if torch.any(mask):
            tfm_batch = self.apply_mask(tfm_batch, mask)
            output = self._transform(tfm_batch)
            for k in self.signal_keys:
                batch[k][mask] = output[k]

        return batch

    def _transform(self, batch):
        for transform in self.transforms:
            batch = transform(batch)
        return batch

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        parameters = {}
        for transform in self.transforms:
            parameters.update(transform.instantiate(state, signal=signal))
        return parameters


class Choose(Compose):
    # Class logic is the same as Compose, but instead of applying all
    # the transforms in sequence, it applies just a single transform,
    # which is picked deterministically by summing all of the `random_state`
    # integers (which could be just one or a batch of integers), and then
    # using the sum as a seed to build a RandomState object that it then
    # calls `choice` on, with probabilities `self.weights``.
    def __init__(
        self,
        transforms: list,
        weights: list = None,
        max_seed: int = 1000,
        prob: float = 1.0,
    ):
        super().__init__(transforms, prob=prob)
        self.keys.append("random_state")

        if weights is None:
            _len = len(self.transforms)
            weights = [1 / _len for _ in range(_len)]
        self.weights = np.array(weights)
        self.max_seed = max_seed

    def _transform(self, batch):
        state = batch["random_state"].sum().item()
        state = util.random_state(state)
        transform = state.choice(self.transforms, p=self.weights)
        return transform(batch)

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        parameters = super()._instantiate(state, signal)
        parameters["random_state"] = state.randint(self.max_seed)
        return parameters


class ClippingDistortion(BaseTransform):
    def __init__(
        self,
        perc: tuple = ("uniform", 0.0, 0.1),
        signal_keys: list = None,
        prob: float = 1.0,
    ):
        keys = ["perc"]
        super().__init__(keys=keys, signal_keys=signal_keys, prob=prob)

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
        signal_keys: list = None,
        prob: float = 1.0,
    ):
        keys = ["eq"]
        super().__init__(keys=keys, signal_keys=signal_keys, prob=prob)

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
        signal_keys: list = None,
        prob: float = 1.0,
    ):
        keys = ["channels"]
        super().__init__(keys=keys, signal_keys=signal_keys, prob=prob)

        self.channels = channels

    def _instantiate(self, state: RandomState):
        return {"channels": util.sample_from_dist(self.channels, state)}

    def _transform(self, signal, channels):
        return signal.quantization(channels)


class MuLawQuantization(BaseTransform):
    def __init__(
        self,
        channels: tuple = ("choice", [8, 32, 128, 256, 1024]),
        signal_keys: list = None,
        prob: float = 1.0,
    ):
        keys = ["channels"]
        super().__init__(keys=keys, signal_keys=signal_keys, prob=prob)

        self.channels = channels

    def _instantiate(self, state: RandomState):
        return {"channels": util.sample_from_dist(self.channels, state)}

    def _transform(self, signal, channels):
        return signal.mulaw_quantization(channels)


class BackgroundNoise(BaseTransform):
    def __init__(
        self,
        snr: tuple = ("uniform", 10.0, 30.0),
        csv_files: List[str] = None,
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 3,
        signal_keys: list = None,
        prob: float = 1.0,
    ):
        """
        min and max refer to SNR.
        """
        keys = ["eq", "snr", "bg_signal"]
        super().__init__(keys=keys, signal_keys=signal_keys, prob=prob)

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

    def _transform(self, signal, bg_signal, snr, eq):
        # Clone bg_signal so that transform can be repeatedly applied
        # to different signals with the same effect.
        return signal.mix(bg_signal.clone(), snr, eq)


class RoomImpulseResponse(BaseTransform):
    def __init__(
        self,
        drr: tuple = ("uniform", 0.0, 30.0),
        csv_files: List[str] = None,
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 6,
        signal_keys: list = None,
        prob: float = 1.0,
    ):
        keys = ["eq", "drr", "ir_signal"]
        super().__init__(keys=keys, signal_keys=signal_keys, prob=prob)

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

    def _transform(self, signal, ir_signal, drr, eq):
        # Clone ir_signal so that transform can be repeatedly applied
        # to different signals with the same effect.
        return signal.apply_ir(ir_signal.clone(), drr, eq)


class VolumeChange(BaseTransform):
    def __init__(
        self,
        db: tuple = ("uniform", -12.0, 0.0),
        signal_keys: list = ["signal", "original"],
        prob: float = 1.0,
    ):
        keys = ["db"]
        super().__init__(keys=keys, signal_keys=signal_keys, prob=prob)
        self.db = db

    def _instantiate(self, state: RandomState):
        return {"db": util.sample_from_dist(self.db, state)}

    def _transform(self, signal, db):
        return signal.volume_change(db)


class VolumeNorm(BaseTransform):
    def __init__(
        self,
        db: float = -24,
        signal_keys: list = ["signal", "original"],
        prob: float = 1.0,
    ):
        keys = ["loudness"]
        super().__init__(keys=keys, signal_keys=signal_keys, prob=prob)

        self.db = db

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        return {"loudness": signal.metadata["file_loudness"]}

    def _transform(self, signal, loudness):
        db_change = self.db - loudness
        return signal.volume_change(db_change)


class Silence(BaseTransform):
    def __init__(self, signal_keys: list = ["signal", "original"], prob: float = 0.1):
        super().__init__(signal_keys=signal_keys, prob=prob)

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
        signal_keys: list = None,
        prob: float = 1,
    ):
        keys = ["cutoff"]
        super().__init__(keys=keys, signal_keys=signal_keys, prob=prob)

        self.cutoff = cutoff

    def _instantiate(self, state: RandomState):
        return {"cutoff": util.sample_from_dist(self.cutoff, state)}

    def _transform(self, signal, cutoff):
        return signal.low_pass(cutoff)


class HighPass(BaseTransform):
    def __init__(
        self,
        cutoff: tuple = ("choice", [50, 100, 250, 500, 1000]),
        signal_keys: list = None,
        prob: float = 1,
    ):
        keys = ["cutoff"]
        super().__init__(keys=keys, signal_keys=signal_keys, prob=prob)

        self.cutoff = cutoff

    def _instantiate(self, state: RandomState):
        return {"cutoff": util.sample_from_dist(self.cutoff, state)}

    def _transform(self, signal, cutoff):
        return signal.high_pass(cutoff)


class RescaleAudio(BaseTransform):
    def __init__(self, val: float = 1.0, prob: float = 1, signal_keys: list = None):
        super().__init__(signal_keys=signal_keys, prob=prob)

        self.val = val

    def _transform(self, signal):
        return signal.ensure_max_of_audio(self.val)
