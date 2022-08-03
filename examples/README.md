
More examples are provided at [mdpi2022](https://github.com/mlzxy/mdpi2022).

---
# MNIST Layerwise Pruning + Quantization Example

To train a full precision network:

```bash
python3 examples/mnist.py
# Test set: Average loss: 0.0234, Accuracy: 9922/10000 (99%)
```

To train an 4-bit quantized and 75% pruned network:

```bash
python3 examples/mnist.py --pq
# Test set: Average loss: 0.0270, Accuracy: 9911/10000 (99%)
```
