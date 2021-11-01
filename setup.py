"""setup.py: setuptools control."""

import os
import re
from setuptools import Extension, find_packages, setup


# copied from https://stackoverflow.com/a/21784019/6238109
def get_version(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(here, filename))
    version_file = f.read()
    f.close()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


__version__ = get_version("qsparse/__init__.py")


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
    url="https://qsparse.readthedocs.io/",
    download_url="https://github.com/mlzxy/qsparse/tags",
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
