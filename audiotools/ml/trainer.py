"""
A trainer class built on Pytorch Ignite with
progress bars from Rich. Requirements:
"""
import contextlib
import heapq
import time
from collections import defaultdict
from datetime import timedelta

import ignite
import torch
import torchmetrics
from ignite.engine.events import Events
from ignite.handlers import TerminateOnNan
from ignite.handlers import Timer as IgniteTimer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter


def iter_summary(output, width=None) -> Table:
    """Make a table summarizing each iteration."""
    table = Table(expand=True, width=width)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="bright_blue")
    table.add_column("Smoothed", style="bright_green")

    for k, v in output.items():
        table.add_row(k, f"{v['value']:7.4f}", f"{v['smoothed']:7.4f}")
    return table


def epoch_summary(epoch, output, width=None) -> Table:
    """Make a table summarizing each epoch."""
    if width is None:
        expand = False
    else:
        expand = True
    table = Table(title=f"Summary for Epoch {epoch}", expand=expand, width=width)

    table.add_column("Key", style="cyan", justify="right")
    table.add_column("Value", style="green")

    for k, v in output.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:5.2f}")
        else:
            table.add_row(k, v)

    return table


class SimpleTimer:
    def __init__(self):
        self.start_time = time.time()

    def __call__(self, message=None):
        time_taken = time.time() - self.start_time
        if message is not None:
            print(f"{time_taken:1.4f}s: {message}")
        return time_taken


class BaseTrainer:
    def __init__(
        self,
        writer: SummaryWriter = None,
        width: int = 87,
        refresh_rate: float = 1.0,
        rank=0,
        quiet: bool = False,
        record_memory: bool = False,
        **kwargs,
    ):
        """
        Sub-class this and implement three functions:

        1. train_loop: The training loop.
        2. val_loop: The validation loop.
        3. checkpoint: What to checkpoint at the end of each epoch.
           Use self.is_best(engine, metric_you_care_about) to figure out
           if the model should be saved to latest or best. Or use
           self.top_k(engine, metric_you_care_about, k) to see if the
           model you're saving is in the Top K of all models so far.

        Note that if you're in PyCharm, you'll need to do this to
        get it to work: "PyCharm users will need to enable “emulate terminal”
        in output console option in run/debug configuration to see
        styled output." (from Rich docs).

        If you're using this with distributed = True, then only the
        rank 0 process will do logging. Additionally, self.val_loop
        and self.checkpoint will only run in the rank 0 process.

        Example
        -------
        >>> model = ...
        >>> train_data = ...
        >>> val_data = ...
        >>> tb = SummaryWriter(...)
        >>>
        >>> class Trainer(BaseTrainer):
        >>>     def train_loop(self, engine, batch):
        >>>         batch = self.prepare_batch(batch)
        >>>         output = model(batch)
        >>>         sleep(0.01)
        >>>         return {"l1": np.random.randn(), "mse": np.random.randn()}
        >>>
        >>>     def val_loop(self, engine, batch):
        >>>         batch = self.prepare_batch(batch)
        >>>         output = model(batch)
        >>>         sleep(0.01)
        >>>         return {"l1": np.random.randn(), "mse": np.random.randn()}
        >>>
        >>>     def checkpoint(self, engine):
        >>>         epoch = engine.state.epoch
        >>>         model.save('checkpoints/latest.model.pth')
        >>>         if self.is_best(engine, 'mse/val'):
        >>>             model.save('checkpoints/best.model.pth')
        >>>         if self.top_k(engine, 'mse/val', 5):
        >>>             model.save(f'checkpoints/top{k}.{epoch}.model.pth')
        >>>
        >>> trainer = Trainer(writer=tb)
        >>> trainer.run(train_data, val_data, num_epochs=3)

        Parameters
        ----------
        writer : SummaryWriter, optional
            Writer for Tensorboard. If this is not the rank 0 process, and
            distributed is False, it will be set to None, by default None.
        width : int, optional
            Width of the tables to render, by default 87
        refresh_rate : int, optional
            Refresh rate of progress bars, by default 10
        rank : int, optional
            The rank of the local process, by default 0.
        quiet : bool, optional
            Whether or not to show progress bars during training, by default
            False.
        """
        self.width = width
        self.refresh_rate = refresh_rate
        self.rank = rank
        self.writer = writer if rank == 0 else None
        self.quiet = quiet
        self.record_memory = record_memory

        self.pbar = Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            "{task.completed}/{task.total}",
            BarColumn(),
            TimeElapsedColumn(),
            "/",
            TimeRemainingColumn(),
        )
        self.live = self.pbar
        self.epoch_summary = None
        self.log_file = None

        # Set up trainer engine
        self.trainer = ignite.engine.Engine(self._train_loop)
        self.trainer.state.logs = {
            "epoch": defaultdict(lambda: []),
        }
        self.trainer.state.prefix = "train"
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self.collect_metrics)
        self.trainer.add_event_handler(Events.EPOCH_STARTED, self.before_epoch)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.validate)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.after_epoch)

        # Set up validator engine
        self.val_data = None
        self.validator = ignite.engine.Engine(self._val_loop)
        self.validator.state.logs = {
            "epoch": defaultdict(lambda: []),
        }
        self.validator.state.prefix = "val"
        self.validator.add_event_handler(
            Events.ITERATION_COMPLETED, self.collect_metrics
        )

        for k, v in kwargs.items():
            if hasattr(self, k):
                raise ValueError(f"{k} set in kwargs but overwrites self.{k}.")
            setattr(self, k, v)

        f = lambda: {
            "smoothed": torchmetrics.MeanMetric(),
            "value": torchmetrics.MeanMetric(),
        }
        self.metrics = defaultdict(lambda: defaultdict(f))

        if self.rank == 0:
            self.trainer.add_event_handler(
                Events.EPOCH_COMPLETED, self.on_epoch_completed
            )
            self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.checkpoint)
            self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self.log_metrics)
            self.trainer.add_event_handler(
                Events.ITERATION_COMPLETED, self.update_progress
            )

            # Setting up timers
            self.epoch_timer = IgniteTimer()
            self.epoch_timer.attach(
                self.trainer, start=Events.EPOCH_STARTED, step=Events.EPOCH_COMPLETED
            )
            self.overall_timer = IgniteTimer()
            self.overall_timer.attach(
                self.trainer, start=Events.STARTED, pause=Events.COMPLETED
            )

            # Setting up validation engine
            self.validator.add_event_handler(
                Events.ITERATION_COMPLETED, self.log_metrics
            )
            self.validator.add_event_handler(
                Events.ITERATION_COMPLETED, self.update_progress
            )

            self.stdout_console = Console()
            self.file_console = Console(file=open("log.txt", "w"))

    @property
    def state(self):
        return self.trainer.state

    def state_dict(self):
        trainer_state_dict = self.trainer.state_dict()

        def _to_dict(x):
            return {k: dict(v) for k, v in x.items()}

        trainer_state_dict["logs"] = _to_dict(self.trainer.state.logs)
        validator_state_dict = self.validator.state_dict()
        validator_state_dict["logs"] = _to_dict(self.validator.state.logs)

        return {
            "trainer": trainer_state_dict,
            "validator": validator_state_dict,
        }

    def load_state_dict(self, state_dict):
        self.trainer.state.logs = state_dict["trainer"].pop("logs")
        self.trainer.load_state_dict(state_dict["trainer"])

        if self.rank == 0:
            self.validator.state.logs = state_dict["validator"].pop("logs")
            self.validator.load_state_dict(state_dict["validator"])
        return

    def save(self, save_path, **kwargs):
        state = {"state_dict": self.state_dict()}
        state.update(kwargs)
        torch.save(state, save_path)
        return save_path

    @staticmethod
    def prepare_batch(batch, device="cpu"):
        if isinstance(batch, dict):
            for key, val in batch.items():
                try:
                    batch[key] = val.to(device).float()
                except:
                    pass
        elif torch.is_tensor(batch):
            batch = batch.to(device).float()
        elif isinstance(batch, list):
            for i in range(len(batch)):
                try:
                    batch[i] = batch[i].to(device).float()
                except:
                    pass
        return batch

    def wrapper(self, loop, engine, batch):
        timer = SimpleTimer()

        time_outside_loop = 0.0
        if hasattr(self, "timer_outside_loop"):
            time_outside_loop = self.timer_outside_loop()

        output = loop(engine, batch)

        output["perf/time_per_step"] = timer()
        output["perf/time_outside_step"] = time_outside_loop

        if self.record_memory:
            factor = 1024**3
            output["perf/memory_allocated"] = torch.cuda.memory_allocated() / factor
            output["perf/max_memory_allocated"] = (
                torch.cuda.max_memory_allocated() / factor
            )
            output["perf/memory_reserved"] = torch.cuda.memory_reserved() / factor
            output["perf/max_memory_reserved"] = (
                torch.cuda.max_memory_reserved() / factor
            )

        self.timer_outside_loop = SimpleTimer()

        return output

    def _train_loop(self, engine, batch):
        output = self.wrapper(self.train_loop, engine, batch)
        return output

    def _val_loop(self, engine, batch):
        output = self.wrapper(self.val_loop, engine, batch)
        return output

    def train_loop(self, engine, batch):
        """
        Performs a single training step of the model
        given the batch. Use self.prepare_batch to
        move the batch to the desired device, or
        override it and write your own. self.prepare_batch
        expects that your batches are dictionaries of
        tensors.

        Parameters
        ----------
        engine : ignite.engine.Engine
            PyTorch Ignite engine. Access engine.state to find
            out things about the engine's current state (e.g.
            engine.state.iteration, engine.state.epoch).
        batch : dict, list
            A data batch produced by your dataloader.

        Returns
        ------
        dict
            Return a dictionary of metrics (e.g. l1 loss), with
            keys corresponding to the metric name and values
            corresponding to Tensors of size 1 containing the
            value of that metric. .item() will be called on each
            tensor, and the values will be logged to the live
            progress view as well as Tensorboard if self.writer
            is not None.
        """
        raise NotImplementedError()

    def val_loop(self, engine, batch):
        """
        Performs a single validation step of the model
        given the batch. Use self.prepare_batch to
        move the batch to the desired device, or
        override it and write your own. self.prepare_batch
        expects that your batches are dictionaries of
        tensors.

        Parameters
        ----------
        engine : ignite.engine.Engine
            PyTorch Ignite engine. Access engine.state to find
            out things about the engine's current state (e.g.
            engine.state.iteration, engine.state.epoch).
        batch : dict, list
            A data batch produced by your dataloader.

        Returns
        ------
        dict
            Return a dictionary of metrics (e.g. l1 loss), with
            keys corresponding to the metric name and values
            corresponding to Tensors of size 1 containing the
            value of that metric. .item() will be called on each
            tensor, and the values will be logged to the live
            progress view as well as Tensorboard if self.writer
            is not None.
        """
        raise NotImplementedError()

    def checkpoint(self, engine):
        """
        This callback is called after Events.EPOCH_COMPLETED.
        You can use it to save models, optimizer, and whatever
        else you want to do. Use with self.is_best and self.top_k
        to figure out if the model is the best model according to
        some metric (e.g. validation loss), or in the top K models
        so far. You can use these to decide where to save the model
        (e.g. save the model to latest.pth vs best.pth, or top3.pth).

        Parameters
        ----------
        engine : ignite.engine.Engine
            PyTorch Ignite engine. Access engine.state to find
            out things about the engine's current state (e.g.
            engine.state.iteration, engine.state.epoch).
        """
        pass

    def after_epoch(self, engine):
        """
        This callback is called after Events.EPOCH_COMPLETED, but
        in every process (if you are using DDP).
        """
        pass

    def before_epoch(self, engine):
        """
        This callback is called after Events.EPOCH_STARTED, but
        in every process (if you are using DDP). It can allow you to implement
        a curriculum however you see fit - for example check the epoch or
        iteration number, and change the audio duration during training from
        0.5 seconds to 1.0 seconds, or do other things before each epoch.

        Parameters
        ----------
        engine : ignite.engine.Engine
            PyTorch Ignite engine. Access engine.state to find
            out things about the engine's current state (e.g.
            engine.state.iteration, engine.state.epoch).
        """
        pass

    @staticmethod
    def is_best(engine, loss_key, minimize=True):
        """
        Is this the best model so far, according to loss_key?
        """
        loss_vals = engine.state.logs["epoch"][loss_key]
        fn = min if minimize else max
        return loss_vals[-1] == fn(loss_vals)

    @staticmethod
    def top_k(engine, loss_key, k, minimize=True):
        """
        Is this model in the top K models so far, according
        to loss_key?
        """
        loss_vals = engine.state.logs["epoch"][loss_key]
        fn = heapq.nsmallest if minimize else heapq.nlargest
        top_k = fn(k, loss_vals)
        model_rank = None
        for i, x in enumerate(top_k):
            if loss_vals[-1] == x:
                model_rank = i
                break
        return model_rank

    def print(self, *args, **kwargs):
        if self.rank == 0:
            self.live.console.print(*args, **kwargs)
            self.file_console.print(*args, **kwargs)
        else:
            print(*args)

    def validate(self, engine):
        if self.val_data is not None:
            self.validator.run(self.val_data)

        for _, metrics in self.metrics.items():
            for k, v in metrics.items():
                v["smoothed"].reset()
                v["value"].reset()

    def on_epoch_completed(self, engine):
        def _summarize_metrics(_engine):
            for k, v in _engine.state.output.items():
                k_ = f"{k}/{_engine.state.prefix}"
                if k_ not in _engine.state.logs["epoch"]:
                    _engine.state.logs["epoch"][k_] = []
                _engine.state.logs["epoch"][k_].append(v["smoothed"])
            return _engine.state.logs["epoch"]

        _summarize_metrics(self.trainer)

        if self.val_data is not None:
            val_epoch_metrics = _summarize_metrics(self.validator)
            self.trainer.state.logs["epoch"].update(val_epoch_metrics)

        summary = {}
        for k, v in self.trainer.state.logs["epoch"].items():
            summary[k] = v[-1]
            if self.writer is not None:
                self.writer.add_scalar(k, v[-1], engine.state.epoch)

        summary["Epoch took"] = str(timedelta(seconds=self.epoch_timer.value()))
        summary["Time since start"] = str(timedelta(seconds=self.overall_timer.value()))

        self.epoch_summary = epoch_summary(engine.state.epoch, summary, self.width)
        self.print(self.epoch_summary)

        if not self.quiet:
            self.pbar.reset(self.trainer.state.pbar)
            self.pbar.advance(self.epoch_pbar)
            if self.val_data is not None:
                self.pbar.reset(self.validator.state.pbar)

    def collect_metrics(self, engine):
        output = engine.state.output
        prefix = engine.state.prefix
        metrics = self.metrics[prefix]

        device = "cpu"
        for k, v in output.items():
            if hasattr(v, "device"):
                device = v.device
                break

        for k, v in output.items():
            if not torch.is_tensor(v):
                v = torch.FloatTensor([v]).to(device)
            v = v.detach()
            metrics[k]["smoothed"].to(v.device).update(v)
            metrics[k]["value"].to(v.device).update(v)
            output[k] = v.item()

        output = {}
        for k, v in metrics.items():
            output[k] = {
                "value": v["value"].compute().item(),
                "smoothed": v["smoothed"].compute().item(),
            }
            v["value"].reset()

        engine.state.output = output

    def log_metrics(self, engine):
        iteration = engine.state.iteration
        output = engine.state.output

        for k, v in output.items():
            if self.writer is not None:
                if engine.state.prefix != "val":
                    k_ = f"{k}/iter.{engine.state.prefix}"
                    self.writer.add_scalar(k_, v["value"], iteration)

    def update_progress(self, engine):
        if self.quiet:
            return

        iter_table = iter_summary(engine.state.output, self.width)
        self.pbar.advance(engine.state.pbar)

        updated_table = Table.grid()
        updated_table.add_row(iter_table)
        updated_table.add_row(
            Panel.fit(self.pbar, title="[b]Progress", border_style="blue"),
        )
        self.live.update(updated_table)

    def view(self, epoch_length, val_epoch_length, num_epochs):
        if self.rank == 0:
            self.epoch_pbar = self.pbar.add_task(
                "[white]Epoch",
                total=num_epochs,
                completed=self.trainer.state.epoch,
            )
            self.trainer.state.pbar = self.pbar.add_task(
                "[white]Iteration (train)", total=epoch_length
            )
            if val_epoch_length is not None:
                self.validator.state.pbar = self.pbar.add_task(
                    "[white]Iteration (val)", total=val_epoch_length
                )
            return Live(
                refresh_per_second=self.refresh_rate, console=self.stdout_console
            )
        else:
            return contextlib.nullcontext()

    def run(
        self,
        train_data,
        val_data=None,
        num_epochs=None,
        epoch_length=None,
        detect_anomaly=False,
        **kwargs,
    ):
        self.train_data = train_data
        self.val_data = val_data

        epoch_length = len(train_data) if epoch_length is None else epoch_length
        val_epoch_length = len(val_data) if val_data is not None else None

        ctx = self.view(epoch_length, val_epoch_length, num_epochs)
        with torch.autograd.set_detect_anomaly(detect_anomaly):
            with ctx as self.live:
                self.trainer.run(
                    self.train_data,
                    max_epochs=num_epochs,
                    epoch_length=epoch_length,
                    **kwargs,
                )
