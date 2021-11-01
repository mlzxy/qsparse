"""
This file is used to execute pre-build tasks for readthedocs compilation
"""

import subprocess
import shlex


def shell(cmd):
    subprocess.call(shlex.split(cmd))


if __name__ == "__main__":
    shell("mkdir -p site")
    shell("cp ./LICENSE.txt site/")
    shell("cp -r ./docs site/")
    shell("python3 -m pip install ipykernel")
    shell("python3 -m ipykernel install --user")
