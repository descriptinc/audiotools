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
from rich.text import Text
from rich.tree import Tree

from .layers.base import BaseModel


def convert_to_tree(d, tree: Tree):
    for k in d:
        if not isinstance(d[k], dict):
            prefix = "✅ " if d[k] else "❌ "
            style = "green" if d[k] else "red"
            tree.add(Text(prefix + k, style=style))
        else:
            convert_to_tree(d[k], tree.add(k))
    return tree


class BaseModelRegistry:
    def __init__(
        self,
        location: str,
        cache: str = None,
    ):
        cache = location if cache is None else cache
        self.location = str(location)
        self.cache = Path(cache)

    def copy(self, src, dst):  # pragma: no cover
        raise NotImplementedError()

    def upload_model(
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

    def upload(
        self,
        local_path: str,
        domain: str,
        path: str,
    ):
        remote_path = f"{self.location}/{domain}/{path}"
        self.copy(local_path, remote_path)

    def download(
        self,
        domain: str,
        path: str,
        overwrite: bool = False,
    ):
        # Check if model exists locally.
        local_path = self.cache / domain / path
        remote_path = f"{self.location}/{domain}/{path}"
        local_path.parent.mkdir(exist_ok=True, parents=True)

        if not local_path.exists() or overwrite:
            self.copy(remote_path, local_path)

        return local_path

    def get_files(
        self,
        domain: str,
    ):  # pragma: no cover
        raise NotImplementedError()

    def list_models(self, domain: str):
        files = self.get_files(domain)

        def exists(f):
            local_path = self.cache / str(f).split(self.location)[-1]
            return local_path.exists()

        _files = unflatten({str(f): exists(f) for f in files}, splitter="path")
        tree = convert_to_tree(_files, Tree(self.location))
        rich.print(tree)
        return [str(f).split(domain + "/")[-1] for f in files]


class LocalModelRegistry(BaseModelRegistry):
    def copy(self, src, dst):
        Path(dst).parent.mkdir(exist_ok=True, parents=True)
        command = f"cp -r {str(src)} {str(dst)}"
        print(command)
        subprocess.check_call(shlex.split(command))

    def get_files(self, domain: str):
        base_path = f"{self.location}/{domain}"
        files = glob.glob(f"{base_path}/**", recursive=True)
        files = [Path(f).relative_to(self.location) for f in files if Path(f).is_file()]
        return files


class GCPModelRegistry(BaseModelRegistry):  # pragma: no cover
    def copy(self, src, dst):
        command = f"gsutil -m cp -r {str(src)} {str(dst)}"
        print(f"Running {command}")
        subprocess.check_call(shlex.split(command))

    def get_files(self, domain: str):
        base_path = f"{self.location}/{domain}"
        command = f"gsutil ls {base_path}/**"

        files = (
            subprocess.check_output(shlex.split(command)).decode("utf-8").splitlines()
        )
        files = [Path(f).relative_to(self.location) for f in files]
        return files
