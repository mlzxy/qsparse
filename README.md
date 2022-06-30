# QSPARSE

![](docs/assets/coverage.svg)   [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

QSPARSE provides the open source implementation of the quantization and pruning methods proposed in our MDPI2022 submission. This library was developed to support and demonstrate strong performance among various computer vision experiments mentioned in our paper.

<table>
<tr>
<th>Full Precision</th>
<th>Joint Quantization <sub style="font-size:8px">8bit</sub> and Pruning <sub style="font-size:8px">50%</sub> </th>
</tr>
<tr>
<td >

```python
import torch.nn as nn

net = nn.Sequential(
    nn.Conv2d(3, 32, 5),
    nn.ReLU(),
    nn.ConvTranspose2d(32, 3, 5, stride=2),
)
```

</td>
<td>

```python
import torch.nn as nn
from qsparse import prune, quantize

net = nn.Sequential(
    quantize(bits=8),  # input quantization
    quantize(prune(nn.Conv2d(3, 32, 5), 0.5), 8),  # weight pruning+quantization
    prune(sparsity=0.5),  # activation pruning
    quantize(bits=8),  # activation quantization
    nn.ReLU(),
    quantize(prune(nn.ConvTranspose2d(32, 3, 5, stride=2), 0.5), 8),
    quantize(bits=8),
)
```

</td>
</tr>
</table>

It can be seen from the above snippet that our library provides a much simpler and more flexible software interface comparing to existing solutions, e.g. [torch.nn.qat](https://pytorch.org/docs/stable/torch.nn.qat.html). More specifically, our library is **layer-agnostic** and can work with **any** PyTorch module as long as their parameters can be accessed from their `weight` attribute, as is standard practice.


## Installation

QSPARSE can be installed from [PyPI](https://pypi.org/project/qsparse):

```bash
pip install qsparse
```


## Usage

Documentation can be accessed from [Read the Docs](https://qsparse.readthedocs.io/en/latest/).

## Contribute

The development environment can be setup as (Python >= 3.6 is required):

```bash
git clone https://github.com/mlzxy/qsparse
cd qsparse
make dependency
pre-commit install
```

Feel free to raise an [issue](https://github.com/mlzxy/qsparse/issues/new) if you have any questions.


```
