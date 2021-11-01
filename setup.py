"""setup.py: setuptools control."""

import os
import re
import socket
from setuptools import Extension, find_packages, setup
from qsparse import __version__


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
    version=__version__,
    description="train neural networks with joint quantization and pruning on both weights and activations using any pytorch modules",
    long_description=long_descr,
    long_description_content_type="text/markdown",
    author="Xinyu Zhang, Ian Colbert, Srinjoy Das, Ken Kreutz-Delgado",
    author_email="xiz368@eng.ucsd.edu",
    url="https://github.com/mlzxy/qsparse",
    install_requires=load_requirements(),
    python_requires=">=3.6",
    keywords=[
        "pytorch",
        "quantization",
        "pruning",
        "model compression",
        "neural network",
        "machine learning",
    ],
    include_package_data=True,
)
