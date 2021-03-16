import pyloudnorm
import torchaudio
import numpy as np
import torch
import copy

MIN_LOUDNESS = -120

class Meter(pyloudnorm.Meter):
    """Tensorized version of pyloudnorm.Meter. Works with batched audio tensors.
    """
    @staticmethod
    def apply_filter(filter_stage, data):
        passband_gain = filter_stage.passband_gain

        a_coeffs = torch.from_numpy(filter_stage.a).to(data.device).float()
        b_coeffs = torch.from_numpy(filter_stage.b).to(data.device).float()

        _data = data.permute(0, 2, 1)
        filtered = torchaudio.functional.lfilter(_data, a_coeffs, b_coeffs)
        output = passband_gain * filtered.permute(0, 2, 1)
        return output

    def integrated_loudness(self, data):
        if not torch.is_tensor(data):
            data = torch.from_numpy(data).float()
        else:
            data = data.float()
        
        input_data = copy.copy(data)
        # Data always has a batch and channel dimension.
        # Is of shape (nb, nt, nch)
        if input_data.ndim < 2:
            input_data = input_data.unsqueeze(-1)
        if input_data.ndim < 3: 
            input_data = input_data.unsqueeze(0)

        nb, nt, nch = input_data.shape  

        # Apply frequency weighting filters - account 
        # for the acoustic respose of the head and auditory system
        for _, filter_stage in self._filters.items():
            input_data = self.apply_filter(filter_stage, input_data)

        G = torch.from_numpy(np.array([1.0, 1.0, 1.0, 1.41, 1.41])) # channel gains
        T_g = self.block_size # 400 ms gating block standard
        Gamma_a = -70.0 # -70 LKFS = absolute loudness threshold
        overlap = 0.75 # overlap of 75% of the block duration
        step = 1.0 - overlap # step size by percentage

        unfolded = torch.nn.functional.unfold(
            input_data.permute(0, 2, 1).reshape(nb * nch, 1, 1, nt),
            (1, int(T_g * self.rate)), 
            stride=int(T_g * self.rate * step)
        )
        unfolded = unfolded.squeeze(2).reshape(nb, nch, unfolded.shape[1], unfolded.shape[2])
        z = (1.0 / (T_g * self.rate)) * unfolded.square().sum(2)
        l = -0.691 + 10.0 * torch.log10((G[None, :nch, None] * z).sum(1, keepdim=True))
        l = l.expand_as(z)

        # find gating block indices above absolute threshold
        z_avg_gated = z
        z_avg_gated[l <= Gamma_a] = 0
        masked = l > Gamma_a
        z_avg_gated = z_avg_gated.sum(2) / masked.sum(2)

        # calculate the relative threshold value (see eq. 6)
        Gamma_r = -0.691 + 10.0 * torch.log10((z_avg_gated * G[None, :nch]).sum(-1)) - 10.0
        Gamma_r = Gamma_r[:, None, None]
        Gamma_r = Gamma_r.expand(nb, nch, l.shape[-1])
        
        # find gating block indices above relative and absolute thresholds  (end of eq. 7)
        z_avg_gated = z
        z_avg_gated[l <= Gamma_a] = 0
        z_avg_gated[l <= Gamma_r] = 0
        masked = (l > Gamma_a) * (l > Gamma_r)
        z_avg_gated = z_avg_gated.sum(2) / masked.sum(2)
        z_avg_gated = torch.nan_to_num(z_avg_gated)

        LUFS = -0.691 + 10.0 * torch.log10((G[None, :nch] * z_avg_gated).sum(1))
        return torch.nan_to_num(LUFS, nan=MIN_LOUDNESS)

class LoudnessMixin:
    def loudness(self, filter_class='K-weighting', block_size=0.400):
        """
        Uses pyloudnorm to calculate loudness.
        Implementation of ITU-R BS.1770-4.
        Allows control over gating block size and frequency weighting filters for 
        additional control.
        Measure the integrated gated loudness of a signal.
        
        Uses the weighting filters and block size defined by the meter
        the integrated loudness is measured based upon the gating algorithm
        defined in the ITU-R BS.1770-4 specification. 
        Supports up to 5 channels and follows the channel ordering: 
        [Left, Right, Center, Left surround, Right surround]
        Args:
            filter_class (str):
              Class of weighting filter used.
              - 'K-weighting' (default)
              - 'Fenton/Lee 1'
              - 'Fenton/Lee 2'
              - 'Dash et al.'
            block_size (float):
              Gating block size in seconds. Defaults to 0.400.
        Returns:
            float: LUFS, Integrated gated loudness of the input 
              measured in dB LUFS.
        """
        original_length = self.signal_length
        if self.signal_duration < 0.5:
            pad_len = int((0.5 - self.signal_duration) * self.sample_rate)
            self.zero_pad(0, pad_len)

        # create BS.1770 meter
        meter = Meter(
            self.sample_rate, filter_class=filter_class, block_size=block_size)
        # measure loudness
        loudness = meter.integrated_loudness(self.audio_data.permute(0, 2, 1))
        self.truncate_samples(original_length)
        min_loudness = (
            torch.ones_like(loudness, device=loudness.device) *
            MIN_LOUDNESS
        )
        return torch.maximum(loudness, min_loudness)
