# QSPARSE

![](docs/assets/coverage.svg)   [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

QSPARSE provides the open source implementation of the quantization and pruning methods proposed in [Learning Low-Precision Structured Subnetworks Using Joint Layerwise Channel Pruning and Uniform Quantization](https://www.mdpi.com/2076-3417/12/15/7829). This library was developed to support and demonstrate strong performance and flexibility among various experiments.

<table>
<tr>
<th>Full Precision</th>
<th>Joint Quantization <sub style="font-size:8px">4bit</sub> and Channel Pruning <sub style="font-size:8px">75%</sub> </th>
</tr>
<tr>
<td >

```python
import torch.nn as nn
net = nn.Sequential(
    nn.Conv2d(3, 32, 5),
    nn.ReLU(),
    nn.ConvTranspose2d(32, 3, 5, stride=2)
)
```

</td>
<td>

```python
import torch.nn as nn
from qsparse import prune, quantize, convert
net = nn.Sequential(
    quantize(nn.Conv2d(3, 32, 5), bits=4), 
    nn.ReLU(),
    prune(sparsity=0.75, dimensions={1}), 
    quantize(bits=8),  
    quantize(nn.ConvTranspose2d(32, 3, 5, stride=2), bits=4)
)
# Automatic conversion is available via `convert`.
# Please refer to documentation for more details.
```

</td>
</tr>
</table>


## Installation

QSPARSE can be installed from [PyPI](https://pypi.org/project/qsparse):

```bash
pip install qsparse
```


## Usage

Documentation can be accessed from [Read the Docs](https://qsparse.readthedocs.io/en/latest/).

Examples of applying QSPARSE to different tasks are provided at [examples](https://github.com/mlzxy/qsparse/tree/main/examples) and [mdpi2022](https://github.com/mlzxy/mdpi2022).



## Citing

If you find this open source release useful, please reference in your paper:

> Zhang, X.; Colbert, I.; Das, S. Learning Low-Precision Structured Subnetworks Using Joint Layerwise Channel Pruning and Uniform Quantization. Appl. Sci. 2022, 12, 7829. https://doi.org/10.3390/app12157829

```bibtex
@Article{app12157829,
	AUTHOR = {Zhang, Xinyu and Colbert, Ian and Das, Srinjoy},
	TITLE = {Learning Low-Precision Structured Subnetworks Using Joint Layerwise Channel Pruning and Uniform Quantization},
	JOURNAL = {Applied Sciences},
	VOLUME = {12},
	YEAR = {2022},
	NUMBER = {15},
	ARTICLE-NUMBER = {7829},
	URL = {https://www.mdpi.com/2076-3417/12/15/7829},
	ISSN = {2076-3417}
}
```
