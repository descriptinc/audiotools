from pathlib import Path
from typing import List

import argbind
import torch
import torchaudio
from torch.utils.tensorboard import SummaryWriter

import audiotools
from audiotools import AudioSignal
from audiotools import transforms as tfm

Adam = argbind.bind(torch.optim.Adam, without_prefix=True)


@argbind.bind(without_prefix=True)
class Model(torchaudio.models.DeepSpeech, audiotools.ml.BaseModel):
    def __init__(self, n_feature: int = 80, n_hidden: int = 128, **kwargs):
        super().__init__(n_feature, n_hidden=n_hidden, **kwargs)
        self.n_mels = n_feature

    def forward(self, signal: AudioSignal):
        data = signal.mel_spectrogram(self.n_mels)
        data = data.permute(0, 1, 3, 2)
        logits = super().forward(data)
        return logits.mean(dim=1)


@argbind.bind(without_prefix=True)
def build_dataset(
    sample_rate: int = 44100,
    duration: float = 0.5,
    sources: List[str] = ["tests/audio/spk.csv", "tests/audio/noises.csv"],
):
    num_classes = len(sources)
    transform = tfm.Compose(
        tfm.RoomImpulseResponse(sources=["tests/audio/irs.csv"]),
        tfm.LowPass(prob=0.5),
        tfm.ClippingDistortion(prob=0.1),
    )
    loader = audiotools.datasets.AudioLoader(sources=sources)
    dataset = audiotools.datasets.AudioDataset(
        loader,
        sample_rate,
        duration=duration,
        transform=transform,
    )
    return dataset, num_classes


@argbind.bind(without_prefix=True)
def train(accel, batch_size: int = 16):
    writer = None
    if accel.local_rank == 0:
        writer = SummaryWriter(log_dir="logs/")

    train_data, num_classes = build_dataset()
    train_dataloader = accel.prepare_dataloader(
        train_data, batch_size=batch_size, collate_fn=audiotools.util.collate
    )

    model = accel.prepare_model(Model(n_class=num_classes))
    optimizer = Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    class Trainer(audiotools.ml.BaseTrainer):
        def train_loop(self, engine, batch):
            batch = audiotools.util.prepare_batch(batch, accel.device)

            signal = batch["signal"]
            kwargs = batch["transform_args"]
            signal = train_data.transform(signal.clone(), **kwargs)
            label = batch["source_idx"]

            model.train()
            optimizer.zero_grad()
            logits = model(signal)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            return {"loss": loss}

        def checkpoint(self, engine):
            ckpt_path = Path("checkpoints/")
            metadata = {"logs": dict(engine.state.logs["epoch"])}
            ckpt_path.mkdir(parents=True, exist_ok=True)
            accel.unwrap(model).save(ckpt_path / "latest.model.pth", metadata)

            if self.is_best(engine, "loss/train"):
                accel.unwrap(model).save(ckpt_path / "best.model.pth", metadata)

    trainer = Trainer(writer=writer, rank=accel.local_rank)
    trainer.run(train_dataloader, num_epochs=10, epoch_length=100)


@argbind.bind(without_prefix=True)
def run(runs_dir: str = "/tmp", seed: int = 0, amp: bool = False):
    audiotools.util.seed(seed)
    with audiotools.ml.Accelerator(amp=amp) as accel:
        with audiotools.ml.Experiment(runs_dir) as exp:
            print(f"Switching working directory to {exp.exp_dir}")
            exp.snapshot(lambda f: "tests/audio" in f or "examples" in f)
            train(accel)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        run()
