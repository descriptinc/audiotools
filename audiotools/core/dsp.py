import julius
import numpy as np
import torch
import torchaudio

from . import util


class DSPMixin:
    _original_batch_size = None
    _original_num_channels = None
    _padded_signal_length = None

    def collect_windows(self, window_duration, hop_duration):
        """Function which collects overlapping windows from
        an AudioSignal.

        Args:
            audio_signal (AudioSignal): AudioSignal that windows will be collected over.
            window_duration (float): Length of window in seconds.
            hop_duration (float): How much to shift for each window
                (overlap is window_duration - hop_duration) in seconds.

        Returns:
            AudioSignal: Signal of shape (nb * num_windows, nc, window_length).
        """
        self._original_batch_size = self.batch_size
        self._original_num_channels = self.num_channels

        window_length = int(window_duration * self.sample_rate)
        hop_length = int(hop_duration * self.sample_rate)

        if window_length % hop_length != 0:
            factor = window_length // hop_length
            window_length = factor * hop_length

        self.zero_pad(hop_length, hop_length)
        self._padded_signal_length = self.signal_length

        # self.audio_data: (nb, nch, nt).
        unfolded = torch.nn.functional.unfold(
            self.audio_data.reshape(-1, 1, 1, self.signal_length),
            kernel_size=(1, window_length),
            stride=(1, hop_length),
        )
        # unfolded: (nb * nch, window_length, num_windows).
        # -> (nb * nch * num_windows, 1, window_length)
        unfolded = unfolded.permute(0, 2, 1).reshape(-1, 1, window_length)
        self.audio_data = unfolded
        return self

    def overlap_and_add(self, hop_duration):
        """Function which takes a list of windows and overlap adds them into a
        signal the same length as `audio_signal`.

        Args:
            windows (list): List of audio signal objects containing each window, produced by
                `OverlapAdd.collect_windows`.
            sample_rate (float): Sample rate of audio signal.
            window_duration (float): Length of window in seconds.
            hop_duration (float): How much to shift for each window
                (overlap is window_duration - hop_duration) in seconds.

        Returns:
            AudioSignal: overlap-and-added signal.
        """
        hop_length = int(hop_duration * self.sample_rate)
        window_length = self.signal_length

        nb, nch = self._original_batch_size, self._original_num_channels

        unfolded = self.audio_data.reshape(nb * nch, -1, window_length).permute(0, 2, 1)
        folded = torch.nn.functional.fold(
            unfolded,
            output_size=(1, self._padded_signal_length),
            kernel_size=(1, window_length),
            stride=(1, hop_length),
        )

        norm = torch.ones_like(unfolded, device=unfolded.device)
        norm = torch.nn.functional.fold(
            norm,
            output_size=(1, self._padded_signal_length),
            kernel_size=(1, window_length),
            stride=(1, hop_length),
        )

        folded = folded / norm

        folded = folded.reshape(nb, nch, -1)
        self.audio_data = folded
        self.trim(hop_length, hop_length)
        return self

    def low_pass(self, cutoffs, zeros=51):
        cutoffs = util.ensure_tensor(cutoffs, 2, self.batch_size)
        cutoffs = cutoffs / self.sample_rate
        filtered = torch.empty_like(self.audio_data)
        for i, cutoff in enumerate(cutoffs):
            filtered[i] = julius.lowpass_filter(self.audio_data[i], cutoff, zeros=zeros)
        self.audio_data = filtered
        return self

    def find_shelf(self, thresh=2):
        fft = torch.fft.rfft(self.audio_data, dim=-1)
        psd = fft.abs().pow(2).clamp(1e-8).log10()
        psd = torch.nn.functional.avg_pool1d(psd, kernel_size=3, stride=1)

        psd[psd < thresh] = 0
        psd[psd > thresh] = 1
        diffs = torch.diff(psd).abs()

        vals, bins = diffs.max(dim=-1)
        bins[vals == 0] = self.signal_length
        cutoffs = (self.sample_rate / 2) * (bins + 1) / psd.shape[-1]
        return cutoffs
