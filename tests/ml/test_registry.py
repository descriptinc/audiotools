import datetime
import json
import tempfile
from pathlib import Path

from torch import nn

from audiotools import ml


class Generator(ml.BaseModel):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class Discriminator(ml.BaseModel):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def test_local_registry():
    ml.BaseModel.EXTERN += ["test_registry"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        registry = ml.registry.LocalModelRegistry(tmpdir / "remote", tmpdir / "cache")

        generator = Generator()
        discriminator = Discriminator()

        version = datetime.datetime.now().strftime("%Y%m%d")
        gen_path = registry.upload_model(generator, "domain")
        disc_path = registry.upload_model(discriminator, "domain")

        assert version in gen_path
        assert version in disc_path

        version = "test"
        gen_path = registry.upload_model(generator, "domain", version=version)
        disc_path = registry.upload_model(discriminator, "domain", version=version)

        assert version in gen_path
        assert version in disc_path

        models = registry.list_models("domain")
        for model in models:
            registry.download("domain", model)
        registry.list_models("domain")

        with open(tmpdir / "metadata.json", "w") as f:
            d = {"test": "test"}
            json.dump(d, f)
        registry.upload_file(
            tmpdir / "metadata.json", "domain", f"{version}/metadata/metadata_a.json"
        )
        registry.upload_file(
            tmpdir / "metadata.json", "domain", f"{version}/metadata/metadata_b.json"
        )
        registry.list_models("domain")
        registry.download("domain", f"{version}/metadata")
        registry.list_models("domain")
