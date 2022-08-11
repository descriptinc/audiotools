from pathlib import Path

import torch
import torchaudio

import audiotools
from audiotools import AudioSignal
from audiotools import transforms as tfm


class Dataset(audiotools.datasets.BaseDataset):
    def __init__(
        self,
        sample_rate: int,
        transform,
        n_examples: int = 1000,
        duration: float = 0.5,
        num_channels: int = 1,
    ):
        super().__init__(
            n_examples, duration=duration, transform=transform, sample_rate=sample_rate
        )
        self.num_channels = num_channels

    def __getitem__(self, idx):
        state = audiotools.util.random_state(idx)
        signal = AudioSignal.zeros(self.duration, self.sample_rate, self.num_channels)
        return {
            "idx": idx,
            "signal": signal,
            "transform_args": self.transform.instantiate(state, signal=signal),
        }

    def apply(self, batch):
        signal = batch["signal"]
        kwargs = batch["transform_args"]
        one_hot = kwargs["Choose"]["one_hot"].long()
        output = self.transform(signal.clone(), **kwargs)
        return output, one_hot


class Model(torchaudio.models.DeepSpeech, audiotools.ml.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_mels = self.fc1.fc.in_features

    def forward(self, signal):
        data = signal.mel_spectrogram(self.n_mels)
        data = data.permute(0, 1, 3, 2)
        return super().forward(data)


def train():
    transform = tfm.Choose(
        tfm.Compose(
            tfm.AudioSource(["tests/audio/spk.csv"]),
            tfm.RoomImpulseResponse(["tests/audio/irs.csv"]),
            tfm.LowPass(prob=0.5),
            tfm.ClippingDistortion(prob=0.1),
            name="speech",
        ),
        tfm.Compose(
            tfm.AudioSource(["tests/audio/noises.csv"]),
            tfm.Equalizer(),
            name="noise",
        ),
    )

    dataset = Dataset(44100, transform)
    train_data = torch.utils.data.DataLoader(
        dataset, batch_size=16, collate_fn=audiotools.util.collate
    )

    model = Model(80, 128, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    device = "cpu"
    ckpt_path = Path("scratch/checkpoints/")

    class Trainer(audiotools.ml.BaseTrainer):
        def train_loop(self, engine, batch):
            batch = audiotools.util.prepare_batch(batch, device)
            audio, one_hot = dataset.apply(batch)
            model.train()

            optimizer.zero_grad()
            logits = model(audio)
            loss = criterion(logits, one_hot)
            loss.backward()
            optimizer.step()

            return {"loss": loss}

        def checkpoint(self, engine):
            metadata = {"logs": dict(engine.state.logs["epoch"])}
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save(ckpt_path / "latest.model.pth", metadata)

            if self.is_best(engine, "loss/train"):
                model.save(ckpt_path / "best.model.pth", metadata)

    trainer = Trainer()

    trainer.run(train_data, num_epochs=10, epoch_length=100)


if __name__ == "__main__":
    train()
