import time
from collections import defaultdict
from functools import wraps

import torch
import torchmetrics
from rich.console import Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.rule import Rule
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter

# Progress bars and dashboard


class ProgressTable:
    def __init__(self, tasks):
        self.pbar = Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            "{task.completed}/{task.total}",
            BarColumn(),
            TimeElapsedColumn(),
            "/",
            TimeRemainingColumn(),
        )
        self.tasks = tasks
        for k, v in self.tasks.items():
            v["pbar"] = self.pbar.add_task(**v)
            v["table"] = Table()

    def update(self, key, output, fn_name, advance=1):
        self.pbar.advance(self.tasks[key]["pbar"], advance=advance)
        self.tasks[key]["table"] = self.to_table(output, key)
        tables = [t["table"] for t in self.tasks.values()]
        group = Group(*tables, self.pbar)
        return Group(
            Padding("", (0, 0)),
            Rule(f"[italic]{fn_name}()", style="white"),
            Padding("", (0, 0)),
            Panel.fit(group, padding=(0, 5), title="[b]Progress", border_style="blue"),
        )

    def summary(self, title):
        return Group(Markdown(f"# {title}"), *[t["table"] for t in self.tasks.values()])

    @staticmethod
    def to_table(output, title=None):
        table = Table(title=title, expand=True)
        table.add_column("key", style="cyan")
        table.add_column("value", style="bright_blue")
        for k, v in output.items():
            table.add_row(k, f"{v:10.6f}")
        return table


def progress(progress_bar, live, prefix, check: bool = True, advance: float = 1):
    def decorator(fn):
        @wraps(fn)
        def decorated(*args, **kwargs):
            output = fn(*args, **kwargs)
            assert isinstance(output, dict)
            if check:
                live.update(progress_bar.update(prefix, output, fn.__name__, advance))
            return output

        return decorated

    return decorator


def log_metrics(
    step,
    writer: SummaryWriter,
    prefix: str = "train",
    check: bool = True,
    reset: bool = False,
):
    def decorator(fn):
        @wraps(fn)
        def decorated(*args, **kwargs):
            output = fn(*args, **kwargs)
            assert isinstance(output, dict)
            if check:
                for k, v in output.items():
                    k = f"{k}/{prefix}"
                    writer.add_scalar(k, v, step())
                    if reset and hasattr(v, "reset"):
                        v.reset()
            return output

        return decorated

    return decorator


def track_metrics(smooth: bool = False):
    f = lambda: torchmetrics.MeanMetric()
    metrics = defaultdict(f)

    def decorator(fn):
        @wraps(fn)
        def decorated(*args, **kwargs):
            output = fn(*args, **kwargs)
            assert isinstance(output, dict)

            device = "cpu"
            for k, v in output.items():
                if hasattr(v, "device"):
                    device = v.device
                    break

            for k, v in output.items():
                if not torch.is_tensor(v):
                    v = torch.FloatTensor([v]).to(device)
                v = v.detach()
                metrics[k].to(v.device).update(v)
                output[k] = v.item()

            output = {}
            for k, v in metrics.items():
                output[k] = v.compute().item()
                if not smooth:
                    v.reset()
            return output

        return decorated

    return decorator


def timer(key: str = "perf/time_per_step"):
    def decorator(fn):
        @wraps(fn)
        def decorated(*args, **kwargs):
            s = time.perf_counter()
            output = fn(*args, **kwargs)
            e = time.perf_counter()
            output[key] = e - s
            return output

        return decorated

    return decorator


def when(chk):
    def decorator(fn):
        @wraps(fn)
        def decorated(*args, **kwargs):
            if chk():
                return fn(*args, **kwargs)

        return decorated

    return decorator
