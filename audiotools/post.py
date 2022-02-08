"""Helper script for creating audio-enriched HTML/Discourse posts.
"""
import os
import shlex
import subprocess
from pathlib import Path

import argbind

import audiotools

css = Path(audiotools.__file__).parent / "core" / "templates" / "pandoc.css"


@argbind.bind(without_prefix=True, positional=True)
def create_post(
    in_file: str,
    discourse: bool = False,
):  # pragma: no cover
    env = os.environ.copy()

    if not discourse:
        command = (
            f"codebraid pandoc --from markdown --to html "
            f"--css '{str(css)}' --standalone --wrap=none "
            f"--self-contained --no-cache"
        )
    else:
        env["UPLOAD_TO_DISCOURSE"] = str(int(discourse))
        command = (
            f"codebraid pandoc "
            f"--from markdown --to markdown "
            f"--wrap=none -t gfm --no-cache "
        )

    command += f" {in_file}"
    output = subprocess.check_output(shlex.split(command), env=env)
    print(output.decode(encoding="UTF-8"))


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        create_post()
