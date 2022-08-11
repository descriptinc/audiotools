from pathlib import Path

import torch
import torchaudio
from torch.utils.tensorboard import SummaryWriter

import audiotools
from audiotools import transforms as tfm


class Model(torchaudio.models.DeepSpeech, audiotools.ml.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_mels = self.fc1.fc.in_features

    def forward(self, signal):
        data = signal.mel_spectrogram(self.n_mels)
        data = data.permute(0, 1, 3, 2)
        logits = super().forward(data)
        return logits.mean(dim=1)


def train(accel):
    writer = None
    if accel.local_rank == 0:
        writer = SummaryWriter(log_dir="logs/")

    transform = tfm.Compose(
        tfm.RoomImpulseResponse(csv_files=["tests/audio/irs.csv"]),
        tfm.LowPass(prob=0.5),
        tfm.ClippingDistortion(prob=0.1),
    )
    dataset = audiotools.datasets.CSVDataset(
        44100,
        csv_files=["tests/audio/spk.csv", "tests/audio/noises.csv"],
        duration=0.5,
        transform=transform,
    )
    train_data = accel.prepare_dataloader(
        dataset, batch_size=16, collate_fn=audiotools.util.collate
    )

    model = accel.prepare_model(Model(80, 128, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    device = "cpu"
    ckpt_path = Path("checkpoints/")

    class Trainer(audiotools.ml.BaseTrainer):
        def train_loop(self, engine, batch):
            batch = audiotools.util.prepare_batch(batch, device)

            signal = batch["signal"]
            kwargs = batch["transform_args"]
            signal = dataset.transform(signal.clone(), **kwargs)
            label = batch["label"]

            model.train()

            optimizer.zero_grad()
            logits = model(signal)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            return {"loss": loss}

        def checkpoint(self, engine):
            metadata = {"logs": dict(engine.state.logs["epoch"])}
            ckpt_path.mkdir(parents=True, exist_ok=True)
            accel.unwrap(model).save(ckpt_path / "latest.model.pth", metadata)

            if self.is_best(engine, "loss/train"):
                accel.unwrap(model).save(ckpt_path / "best.model.pth", metadata)

    trainer = Trainer(writer=writer, rank=accel.local_rank)

    trainer.run(train_data, num_epochs=10, epoch_length=100)


if __name__ == "__main__":
    audiotools.util.seed(0)
    with audiotools.ml.Accelerator() as accel:
        with audiotools.ml.Experiment("/tmp") as exp:
            print(f"Switching working directory to {exp.exp_dir}")
            exp.snapshot(lambda f: "tests/audio" in f or "examples" in f)
            train(accel)
