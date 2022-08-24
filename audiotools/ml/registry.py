# Registering models derived from BaseModel in a
# local filestore or in a GCP bucket
import datetime
import glob
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Type

import rich
from flatten_dict import unflatten
from rich.tree import Tree

from . import BaseModel


def convert_to_tree(d, tree: Tree):
    for k in d:
        if d[k] is None:
            tree.add(k)
        else:
            convert_to_tree(d[k], tree.add(k))
    return tree


class BaseModelRegistry:
    def __init__(
        self,
        location: str,
        cache: str = "/tmp/cache",
    ):
        self.location = location
        self.cache = cache

    def copy(self, src, dst):
        raise NotImplementedError()

    def upload(
        self,
        model: Type[BaseModel],
        domain: str,
        version: str = None,
    ):
        model_name = type(model).__name__.lower()
        if version is None:
            version = datetime.datetime.now().strftime("%Y%m%d")
        target_base = f"{self.location}/{domain}/{version}/{model_name}/"
        Path(target_base).mkdir(exist_ok=True, parents=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            package_path = Path(tmpdir) / f"package.pth"
            weights_path = Path(tmpdir) / f"weights.pth"

            model.save(package_path)
            model.save(weights_path, package=False)

            self.copy(package_path, target_base + "package.pth")
            self.copy(weights_path, target_base + "weights.pth")

        return target_base

    def upload_artifact(
        self,
        local_path: str,
        remote_path: str,
    ):
        self.copy(local_path, remote_path)

    def download(
        self,
        domain: str,
        path: str,
        overwrite: bool = False,
    ):
        # Check if model exists locally.
        local_path = Path(self.cache) / domain / path
        remote_path = f"{self.location}/{domain}/{path}"
        local_path.parent.mkdir(exist_ok=True, parents=True)

        if not local_path.exists() or overwrite:
            self.copy(remote_path, local_path)

        return local_path

    def list_models(
        self,
        domain: str,
    ):
        raise NotImplementedError()


class LocalModelRegistry(BaseModelRegistry):
    def copy(self, src, dst):
        Path(dst).parent.mkdir(exist_ok=True, parents=True)
        command = f"cp {str(src)} {str(dst)}"
        subprocess.check_call(shlex.split(command))

    def list_models(self, domain: str):
        base_path = f"{self.location}/{domain}"

        files = glob.glob(f"{self.location}/{domain}/**", recursive=True)
        files = [Path(f).relative_to(base_path) for f in files if Path(f).is_file()]

        files = unflatten({str(f): None for f in files}, splitter="path")
        tree = convert_to_tree(files, Tree(base_path))
        rich.print(tree)


class GCPModelRegistry(BaseModelRegistry):
    def copy(self, src, dst):
        command = f"gsutil -m cp {str(src)} {str(dst)}"
        print(f"Running {command}")
        subprocess.check_call(shlex.split(command))

    def list_models(self, domain: str):
        base_path = f"{self.location}/{domain}"
        command = f"gsutil ls {base_path}/**"

        files = (
            subprocess.check_output(shlex.split(command)).decode("utf-8").splitlines()
        )
        files = [Path(f).relative_to(base_path) for f in files]
        files = unflatten({str(f): None for f in files}, splitter="path")
        tree = convert_to_tree(files, Tree(base_path))
        rich.print(tree)


if __name__ == "__main__":
    from torch import nn

    registry = GCPModelRegistry("gs://wav2wav/debug")

    class Generator(BaseModel):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    class Discriminator(BaseModel):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    generator = Generator()
    discriminator = Discriminator()

    registry.upload(generator, "dummy", version="test")
    registry.upload(discriminator, "dummy", version="test")

    registry.list_models("dummy")
    model_path = registry.download("dummy", "20220824/generator/package.pth")
    model = Generator.load(model_path)
