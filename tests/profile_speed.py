from audiotools import AudioSignal
import audiotools
import tqdm
import torch

class RoomSimulator(torch.utils.data.Dataset):
    def __init__(self, duration):
        super().__init__()
        self.duration = duration

    def __len__(self):
        return 1000*128

    @profile
    def __getitem__(self, idx):
        state = audiotools.util.random_state(idx)
        spk = AudioSignal.excerpt(
            'tests/audio/spk/f10_script4_produced.wav', 
            duration=self.duration, state=state
        )
        nz = AudioSignal.excerpt(
            'tests/audio/nz/f5_script2_ipad_balcony1_room_tone.wav', 
            duration=self.duration, state=state
        )
        ir = AudioSignal('tests/audio/ir/h179_Bar_1txts.wav')

        return {'spk': spk, 'nz': nz, 'ir': ir}

    @profile
    def augment(self, batch, device='cuda:2'):
        with torch.no_grad():
            state = audiotools.util.random_state(None)
            spk_batch, nz_batch, ir_batch = batch['spk'], batch['nz'], batch['ir']

            batch_size = spk_batch.batch_size

            snr = state.uniform(10, 40, batch_size)
            drr = state.uniform(-10, 50, batch_size)

            spk_batch.to(device)
            nz_batch.to(device)
            ir_batch.to(device)

            # Make a copy so we have it later for training targets.
            clean_spk = spk_batch.deepcopy()

            # Augment the noise signal with equalization
            bands = nz_batch.get_bands()
            curve = -1 + 1 * state.rand(nz_batch.batch_size, bands.shape[0])
            nz_batch = nz_batch.equalizer(curve)

            # Augment the impulse response to simulate microphone effects
            # and with varying direct-to-reverberant ratio.
            bands = ir_batch.get_bands()
            curve = -1 + 1 * state.rand(ir_batch.batch_size, bands.shape[0])
            ir_batch = ir_batch.equalizer(curve).alter_drr(drr)

            # Convolve
            noisy_spk = (
                spk_batch
                    .convolve(ir_batch)
                    .mix(nz_batch, snr=snr)
            )
        return {
            'clean': clean_spk, 
            'noisy': noisy_spk
        }

    @staticmethod
    def collate(list_of_dicts):
        dict_of_lists = {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}
        batch = {}
        for k, v in dict_of_lists.items():
            if isinstance(v, list):
                if all(isinstance(s, AudioSignal) for s in v):
                    batch[k] = AudioSignal.batch(v)
                    batch[k].loudness()
        return batch

# We'll apply the following pipeline, randomly getting parameters for each effect.
# 1. Pitch shift
# 2. Time stretch
# 3. Equalize noise.
# 4. Equalize impulse response.
# 5. Convolve speech with impulse response.
# 6. Mix speech and noise at some random SNR. 

@profile
def run(batch_size=128):
    dataset = RoomSimulator(0.5)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=16, batch_size=batch_size,
        collate_fn=RoomSimulator.collate
    )
    pbar = tqdm.trange(len(dataloader))
    for i, batch in enumerate(dataloader):
        batch = dataset.augment(batch)
        pbar.update()

if __name__ == "__main__":
    run()
