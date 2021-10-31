
More examples are provided at [qsparse-examples](https://github.com/mlzxy/qsparse-examples).

---
# MNIST Example

To train a full precision network:

```bash
python3 examples/mnist.py
# Test set: Average loss: 0.0234, Accuracy: 9922/10000 (99%)
```

To train a quantized and pruned network:

```bash
python3 examples/mnist.py --train-mode prune_both-quantize
# Test set: Average loss: 0.0270, Accuracy: 9911/10000 (99%)
```
