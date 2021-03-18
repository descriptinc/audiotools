from audiotools import AudioSignal
import audiotools
import tqdm
import torch

@profile
def load_batch(batch_size, state=None, device='cuda:2'):
    spk_batch = AudioSignal.batch([
        AudioSignal.excerpt('tests/audio/spk/f10_script4_produced.wav', duration=1.0, state=state)
        for _ in range(batch_size)
    ])
    nz_batch = AudioSignal.batch([
        AudioSignal.excerpt('tests/audio/nz/f5_script2_ipad_balcony1_room_tone.wav', duration=1.0, state=state)
        for _ in range(batch_size)
    ])
    ir_batch = AudioSignal.batch([
        AudioSignal('tests/audio/ir/h179_Bar_1txts.wav')
        for _ in range(batch_size)
    ])

    spk_batch.loudness()
    nz_batch.loudness()

    return spk_batch.to(device), nz_batch.to(device), ir_batch.to(device)

# We'll apply the following pipeline, randomly getting parameters for each effect.
# 1. Pitch shift
# 2. Time stretch
# 3. Equalize noise.
# 4. Equalize impulse response.
# 5. Convolve speech with impulse response.
# 6. Mix speech and noise at some random SNR. 

# Seed is given to function for reproducibility.
@profile
def augment(seed, batch_size):
    with torch.no_grad():
        state = audiotools.util.random_state(seed)
        spk_batch, nz_batch, ir_batch = load_batch(batch_size, state)

        n_semitones = state.uniform(-2, 2)
        factor = state.uniform(0.8, 1.2)
        snr = state.uniform(10, 40, batch_size)

        # We're not trying to undo pitch shifting/time streching.
        # spk_batch = (
        #     spk_batch
        #         .pitch_shift(n_semitones)
        #         .time_stretch(factor)
        # )
        # Make a copy so we have it later for training targets.
        clean_spk = spk_batch.deepcopy()

        # Augment the noise signal with equalization
        bands = nz_batch.get_bands()
        curve = -1 + 1 * state.rand(nz_batch.batch_size, bands.shape[0])
        nz_batch = nz_batch.equalizer(curve)

        # Augment the impulse response to simulate microphone effects.
        bands = ir_batch.get_bands()
        curve = -1 + 1 * state.rand(ir_batch.batch_size, bands.shape[0])
        ir_batch = ir_batch.equalizer(curve)

        # Convolve
        noisy_spk = (
            spk_batch
                .convolve(ir_batch)
                .mix(nz_batch, snr=snr)
        )
    return clean_spk, noisy_spk

@profile
def run(batch_size=128):
    for i in tqdm.trange(10):
        augment(i, batch_size)

if __name__ == "__main__":
    run()
