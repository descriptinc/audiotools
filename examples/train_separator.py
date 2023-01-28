from pathlib import Path

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
    sample_rate: int = 8000,
    duration: float = 0.5,
):
    # generate some fake data to train on
    audiotools.util.generate_chord_dataset(max_voices=4, output_dir="chords")

    loaders = {
        f"track_{i}": audiotools.datasets.AudioLoader(
            sources=["chords/track_{i}"],
            transform=tfm.Compose(
                tfm.RoomImpulseResponse(sources=["tests/audio/irs.csv"]),
                tfm.LowPass(prob=0.5),
                tfm.ClippingDistortion(prob=0.1),
                tfm.VolumeNorm(("uniform", -20, -10)),
            ),
        )
        for i in range(4)
    }
    dataset = audiotools.datasets.AudioDataset(
        loaders=loaders,
        sample_rate=sample_rate,
        duration=duration,
        num_channels=1,
    )
    return dataset, list(loaders.keys())


@argbind.bind(without_prefix=True)
def train(accel, batch_size: int = 4):
    writer = None
    if accel.local_rank == 0:
        writer = SummaryWriter(log_dir="logs/")

    train_data, sources = build_dataset()
    train_dataloader = accel.prepare_dataloader(
        train_data, batch_size=batch_size, collate_fn=train_data.collate
    )

    model = accel.prepare_model(Model(num_sources=len(sources)))
    optimizer = Adam(model.parameters())
    criterion = audiotools.metrics.distance.SISDRLoss()

    class Trainer(audiotools.ml.BaseTrainer):
        def train_loop(self, engine, batch):
            batch = audiotools.util.prepare_batch(batch, accel.device)

            for k in sources:
                d = batch[k]
                d["augmented"] = train_data.loaders[k](
                    d["signal"].clone(), **d["transform_args"]
                )

            mixture = sum(batch[k]["augmented"] for k in sources)
            _targets = [batch[k]["signal"] for k in sources]
            targets = mixture.clone()
            targets.audio_data = torch.concat([s.audio_data for s in _targets], dim=-2)

            model.train()
            optimizer.zero_grad()
            estimates = model(mixture)
            loss = criterion(targets, estimates)
            loss.backward()
            optimizer.step()

            # log!
            if engine.state.iteration % 10 == 0:
                mixture.write_audio_to_tb("mixture", writer, engine.state.iteration)
                for i, k in enumerate(sources):
                    batch[k]["signal"].write_audio_to_tb(
                        f"source/{k}", writer, engine.state.iteration
                    )
                    estimates[i].detach().write_audio_to_tb(
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
    trainer.run(train_dataloader, num_epochs=10, epoch_length=10)


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
