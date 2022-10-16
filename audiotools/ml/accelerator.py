import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel

from audiotools.data.datasets import ResumableDistributedSampler as DistributedSampler
from audiotools.data.datasets import ResumableSequentialSampler as SequentialSampler


class Accelerator:  # pragma: no cover
    def __init__(self, amp: bool = False):
        """This class is used to prepare models and dataloaders for
        usage with DDP or DP. Use the functions prepare_model, prepare_dataloader to
        prepare the respective objects. In the case of models, they are moved to
        the appropriate GPU and SyncBatchNorm is applied to them. In the case of
        dataloaders, a sampler is created and the dataloader is initialized with
        that sampler.

        If the world size is 1, prepare_model and prepare_dataloader are
        no-ops. If the environment variable LOCAL_RANK is not set, then the
        script was launched without torch.distributed.run, and DataParallel
        will be used instead of DistributedDataParallel (not recommended), if
        the world size (number of GPUs) is greater than 1.

        Parameters
        ----------
        amp : bool, optional
            Whether or not to enable automatic mixed precision, by default False
        """
        local_rank = os.getenv("LOCAL_RANK", None)
        self.world_size = torch.cuda.device_count()

        self.use_ddp = self.world_size > 1 and local_rank is not None
        self.use_dp = self.world_size > 1 and local_rank is None
        self.device = "cpu" if self.world_size == 0 else "cuda"

        if self.use_ddp:
            local_rank = int(local_rank)
            dist.init_process_group(
                "nccl",
                init_method="env://",
                world_size=self.world_size,
                rank=local_rank,
            )

        self.local_rank = 0 if local_rank is None else local_rank
        self.amp = amp

        class DummyScaler:
            def __init__(self):
                pass

            def step(self, optimizer):
                optimizer.step()

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                return optimizer

            def update(self):
                pass

        self.scaler = torch.cuda.amp.GradScaler() if amp else DummyScaler()
        self.device_ctx = (
            torch.cuda.device(self.local_rank) if torch.cuda.is_available() else None
        )

    def __enter__(self):
        if self.device_ctx is not None:
            self.device_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.device_ctx is not None:
            self.device_ctx.__exit__(exc_type, exc_value, traceback)

    def prepare_model(self, model, **kwargs):
        model = model.to(self.device)
        if self.use_ddp:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(
                model, device_ids=[self.local_rank], **kwargs
            )
        elif self.use_dp:
            model = DataParallel(model, **kwargs)
        return model

    # Automatic mixed-precision utilities
    def autocast(self, *args, **kwargs):
        return torch.cuda.amp.autocast(self.amp, *args, **kwargs)

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def step(self, optimizer):
        self.scaler.step(optimizer)

    def update(self):
        self.scaler.update()

    def prepare_dataloader(self, dataset, start_idx=None, **kwargs):
        if self.use_ddp:
            sampler = DistributedSampler(
                dataset,
                start_idx,
                num_replicas=self.world_size,
                rank=self.local_rank,
            )
            if "num_workers" in kwargs:
                kwargs["num_workers"] = max(kwargs["num_workers"] // self.world_size, 1)
            kwargs["batch_size"] = max(kwargs["batch_size"] // self.world_size, 1)
        else:
            sampler = SequentialSampler(dataset, start_idx)

        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, **kwargs)
        return dataloader

    @staticmethod
    def unwrap(model):
        if hasattr(model, "module"):
            return model.module
        return model
