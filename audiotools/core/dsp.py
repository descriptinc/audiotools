import torchaudio
import torch
import numpy as np
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
            stride=(1, hop_length)
        )
        # unfolded: (nb * nch, window_length, num_windows).
        # -> (nb * nch * num_windows, 1, window_length)
        unfolded = unfolded.permute(0, 2, 1).reshape(
            -1, 1, window_length
        )
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
            stride=(1, hop_length)
        )

        norm = torch.ones_like(unfolded, device=unfolded.device)
        norm = torch.nn.functional.fold(
            norm,
            output_size=(1, self._padded_signal_length),
            kernel_size=(1, window_length),
            stride=(1, hop_length)
        )

        folded = folded / norm

        folded = folded.reshape(nb, nch, -1)
        self.audio_data = folded
        self.trim(hop_length, hop_length)
        return self

    def low_pass(self, cutoffs):
        cutoffs = util.ensure_tensor(cutoffs)
        closest_bins = util.hz_to_bin(
            cutoffs, self.signal_length, self.sample_rate
        )
        fft_data = torch.fft.rfft(self.audio_data)

        filters = torch.zeros(fft_data.shape, device=fft_data.device)
        for i, cutoff in enumerate(closest_bins):
            filters[..., :cutoff] = 1
        filtered_fft = fft_data * filters
        filtered = torch.fft.irfft(filtered_fft)

        self.audio_data = filtered
        return self

    def find_peak_freq(self, freq_min=10000, bump=10):
        stft_data = self.stft()


        # if not self.stft_done:
        #     self.max_frequency_displayed = 20000
        #     return -1

        # elif self.audio_signal_copy.file_name.endswith("wav"):
        #     self.max_frequency_displayed = self.audio_signal_copy.sample_rate // 2
        #     return -1

        # else:
        #     freqs = self.audio_signal_copy.freq_vector
        #     psd = self.audio_signal_copy.get_power_spectrogram_channel(0)
        #     psd = self._log_space_prepare(psd)

        #     freq_bin = self.audio_signal_copy.get_closest_frequency_bin(freq_min)
        #     psd = psd[freq_bin:, :]

        #     # Best candidate
        #     all_power = np.sum(psd, axis=1)
        #     freq_i = np.argmax(np.abs(np.diff(np.log(all_power))))
        #     idx = freq_i + freq_bin + bump
        #     idx = len(freqs) - 1 if idx >= len(freqs) else idx

        #     logger.info(f"Max freq = {freqs[idx]} Hz")

        #     self.max_frequency_displayed = freqs[idx]
        #     return idx
