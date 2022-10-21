from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import argbind
import torch
import torchaudio
from torch.utils.tensorboard import SummaryWriter

import audiotools
from audiotools import AudioSignal
from audiotools import transforms as tfm

Adam = argbind.bind(torch.optim.Adam, without_prefix=True)


@argbind.bind(without_prefix=True)
class Model(torchaudio.models.ConvTasNet, audiotools.ml.BaseModel):
    def __init__(self, num_sources: int = 2, **kwargs):
        super().__init__(num_sources=num_sources, **kwargs)

    def forward(self, signal: AudioSignal):
        sources = super().forward(signal.audio_data)
        return AudioSignal(sources, signal.sample_rate)


@argbind.bind(without_prefix=True)
def build_dataset(
    sample_rate: int = 44100,
    duration: float = 0.5,
    csv_groups: List[str] = [
        {
            "bass": "tests/audio/musdb-7s/bass.csv",
            "drums": "tests/audio/musdb-7s/drums.csv",
        }
    ],
):

    transform = {
        "bass": tfm.Compose(
            tfm.RoomImpulseResponse(csv_files=["tests/audio/irs.csv"]),
            tfm.LowPass(prob=0.5),
            tfm.ClippingDistortion(prob=0.1),
            tfm.VolumeNorm(("uniform", -20, -10)),
        ),
        "drums": tfm.Compose(
            tfm.RoomImpulseResponse(csv_files=["tests/audio/irs.csv"]),
            tfm.ClippingDistortion(prob=0.1),
            tfm.VolumeNorm(("uniform", -20, -10)),
        ),
    }

    dataset = audiotools.datasets.CSVMultiTrackDataset(
        sample_rate=sample_rate,
        csv_groups=csv_groups,
        transform=transform,
        duration=duration,
    )
    return dataset, dataset.source_names


@argbind.bind(without_prefix=True)
def train(accel, batch_size: int = 4):
    writer = None
    if accel.local_rank == 0:
        writer = SummaryWriter(log_dir="logs/")

    train_data, source_names = build_dataset()

    train_dataloader = accel.prepare_dataloader(
        train_data, batch_size=batch_size, collate_fn=audiotools.util.collate
    )

    model = accel.prepare_model(Model(num_sources=len(source_names)))
    optimizer = Adam(model.parameters())
    criterion = audiotools.metrics.spectral.MultiScaleSTFTLoss()

    class Trainer(audiotools.ml.BaseTrainer):
        def train_loop(self, engine, batch):
            batch = audiotools.util.prepare_batch(batch, accel.device)

            signals = batch["signals"]
            tfm_kwargs = batch["transform_args"]
            signals = {
                k: train_data.transform[k](v, **tfm_kwargs[k])
                for k, v in signals.items()
            }

            mixture = sum(signals.values())
            sources = mixture.clone()
            sources.audio_data = torch.concat(
                [s.audio_data for s in signals.values()], dim=-2
            )

            model.train()
            optimizer.zero_grad()
            source_estimates = model(mixture)
            loss = criterion(sources, source_estimates)
            loss.backward()
            optimizer.step()

            # log!
            if engine.state.iteration % 10 == 0:
                mixture.write_audio_to_tb("mixture", writer, engine.state.iteration)
                for i, (k, v) in enumerate(signals.items()):
                    v.write_audio_to_tb(f"source/{k}", writer, engine.state.iteration)
                    source_estimates[i].detach().write_audio_to_tb(
                        f"estimate/{k}", writer, engine.state.iteration
                    )

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
