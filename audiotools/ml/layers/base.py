import inspect
import shutil
import tempfile

import torch
from torch import nn


class BaseModel(nn.Module):
    EXTERN = [
        "audiotools.**",
        "tqdm",
        "__main__",
        "numpy.**",
        "julius.**",
        "torchaudio.**",
    ]
    INTERN = []

    def save(self, path, metadata=None, package=True, intern=[], extern=[], mock=[]):
        sig = inspect.signature(self.__class__)
        args = {}

        for key, val in sig.parameters.items():
            arg_val = val.default
            if arg_val is not inspect.Parameter.empty:
                args[key] = arg_val

        # Look up attibutes in self, and if any of them are in args,
        # overwrite them in args.
        for attribute in dir(self):
            if attribute in args:
                args[attribute] = getattr(self, attribute)

        metadata = {} if metadata is None else metadata
        metadata["kwargs"] = args
        if not hasattr(self, "metadata"):
            self.metadata = {}
        self.metadata.update(metadata)

        if not package:
            state_dict = {"state_dict": self.state_dict(), "metadata": metadata}
            torch.save(state_dict, path)
        else:
            self._save_package(path, intern=intern, extern=extern, mock=mock)

        return path

    @property
    def device(self):
        return list(self.parameters())[0].device

    @classmethod
    def load(cls, location, *args, package_name=None, strict=False, **kwargs):
        try:
            model = cls._load_package(location, package_name=package_name)
        except:
            model_dict = torch.load(location, "cpu")
            metadata = model_dict["metadata"]
            metadata["kwargs"].update(kwargs)

            sig = inspect.signature(cls)
            class_keys = list(sig.parameters.keys())
            for k in list(metadata["kwargs"].keys()):
                if k not in class_keys:
                    metadata["kwargs"].pop(k)

            model = cls(*args, **metadata["kwargs"])
            model.load_state_dict(model_dict["state_dict"], strict=strict)
            model.metadata = metadata

        return model

    def _save_package(self, path, intern=[], extern=[], mock=[], **kwargs):
        package_name = type(self).__name__
        resource_name = f"{type(self).__name__}.pth"

        # Below is for loading and re-saving a package.
        if hasattr(self, "importer"):
            kwargs["importer"] = (self.importer, torch.package.sys_importer)
            del self.importer

        sig = inspect.signature(torch.package.PackageExporter)

        # Why do we use a tempfile, you ask?
        # It's so we can load a packaged model and then re-save
        # it to the same location. torch.package throws an
        # error if it's loading and writing to the same
        # file (this is undocumented).
        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            with torch.package.PackageExporter(f.name, **kwargs) as exp:
                exp.intern(self.INTERN + intern)
                exp.mock(mock)
                exp.extern(self.EXTERN + extern)
                exp.save_pickle(package_name, resource_name, self)

                if hasattr(self, "metadata"):
                    exp.save_pickle(
                        package_name, f"{package_name}.metadata", self.metadata
                    )

            shutil.copyfile(f.name, path)

        # Must reset the importer back to `self` if it existed
        # so that you can save the model again!
        if "importer" in kwargs:
            self.importer = kwargs["importer"][0]
        return path

    @classmethod
    def _load_package(cls, path, package_name=None):
        package_name = cls.__name__ if package_name is None else package_name
        resource_name = f"{package_name}.pth"

        imp = torch.package.PackageImporter(path)
        model = imp.load_pickle(package_name, resource_name, "cpu")
        try:
            model.metadata = imp.load_pickle(package_name, f"{package_name}.metadata")
        except:  # pragma: no cover
            pass
        model.importer = imp

        return model
