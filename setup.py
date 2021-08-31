"""setup.py: setuptools control."""

import os
import re
import socket
import subprocess
from setuptools import Extension, find_packages, setup

version = subprocess.getoutput("git rev-parse HEAD")

with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")


def load_requirements(filename="requirements.txt"):
    with open(filename) as f:
        pkgs = [l.rstrip("\n") for l in f.readlines()]
    return pkgs


pyx_file_list = []
for dirpath, dirnames, files in os.walk("qsparse"):
    for filename in files:
        if filename.endswith(".pyx"):
            pyx_file_list.append(os.path.join(dirpath, filename))
setup(
    name="qsparse",
    packages=find_packages(),
    entry_points={"console_scripts": []},
    version=version,
    description="Joint Quantization & Pruning in PyTorch with Pluggable Primitive API",
    long_description=long_descr,
    author="Xinyu Zhang",
    author_email="xiz368@eng.ucsd.edu",
    url="https://github.com/mlzxy/qsparse",
    install_requires=load_requirements(),
    python_requires=">=3.6",
    include_package_data=True,
)
