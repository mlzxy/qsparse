"""
This file is used to execute pre-build tasks for readthedocs compilation
"""

import subprocess
import shlex


def shell(cmd):
    subprocess.call(shlex.split(cmd))


if __name__ == "__main__":
    shell("mkdir -p _build/html")
    shell("cp ./LICENSE.txt _build/html/")
    shell("cp -r ./docs _build/html/")
    shell("python3 -m pip install ipykernel")
    shell("python3 -m ipykernel install --user")
