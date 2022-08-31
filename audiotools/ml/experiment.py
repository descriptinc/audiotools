"""
Experiment tracking.
"""
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import randomname


class Experiment:
    def __init__(
        self,
        exp_directory: str = "runs/",
        exp_name: str = None,
    ):
        """This class contains utilities for managing experiments.
        It is a context manager, that when you enter it, changes
        your directory to a specified experiment folder (which
        optionally can have an automatically generated experiment
        name, or a specified one), and changes the CUDA device used
        to the specified device (or devices).

        Parameters
        ----------
        exp_directory : str
            Folder where all experiments are saved, by default "runs/".
        exp_name : str, optional
            Name of the experiment, by default uses the current time, date, and
            hostname to save.
        """
        if exp_name is None:
            exp_name = self.generate_exp_name()
        exp_dir = Path(exp_directory) / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.git_tracked_files = (
            subprocess.check_output(
                shlex.split("git ls-tree --full-tree --name-only -r HEAD")
            )
            .decode("utf-8")
            .splitlines()
        )
        self.parent_directory = Path(".").absolute()

    def __enter__(self):
        self.prev_dir = os.getcwd()
        os.chdir(self.exp_dir)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.prev_dir)

    @staticmethod
    def generate_exp_name():
        return randomname.get_name()

    def snapshot(self, filter_fn=lambda f: True):
        """Captures a full snapshot of all the files tracked by git at the time
        the experiment is run. It also captures the diff against the committed
        code as a separate file.
        """
        for f in self.git_tracked_files:
            if filter_fn(f):
                Path(f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.parent_directory / f, f)
