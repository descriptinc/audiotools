from setuptools import find_packages
from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="audiotools",
    version="0.4.6",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Topic :: Artistic Software",
        "Topic :: Multimedia",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Editors",
        "Topic :: Software Development :: Libraries",
    ],
    description="Utilities for handling audio.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Prem Seetharaman, Lucas Gestin",
    author_email="prem@descript.com",
    license="MIT",
    packages=find_packages(),
    package_data={
        "": [
            "core/templates/headers.html",
            "core/templates/widget.html",
            "core/templates/pandoc.css",
        ]
    },
    install_requires=[
        "argbind",
        "numpy",
        "soundfile",
        "pyloudnorm",
        "importlib-resources",
        "scipy",
        "torch",
        "julius",
        "torchaudio",
        "ffmpy",
        "ipython",
        "rich",
        "matplotlib",
        "librosa",
        "pystoi",
        "torch_stoi",
        "flatten-dict",
        "markdown2",
        "pytorch-ignite",
        "randomname",
        # Have to freeze protobuf version, https://github.com/protocolbuffers/protobuf/issues/10051
        # Borrowing pin from tensorboard source: https://github.com/tensorflow/tensorboard/commit/fd4f5ff79374252e313c2e7e9b247bc49ab0d54d.
        "protobuf >= 3.9.2, < 3.20",
        "torchmetrics>=0.7.3",
        "tensorboard",
    ],
    extras_require={
        "tests": [
            "pytest",
            "pytest-cov",
            "line_profiler",
            "tqdm",
            "pesq",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "myst-parser",
            "myst-nb",
            "sphinx-multiversion",
        ],
    },
)
