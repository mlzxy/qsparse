# pytorch-feature-sparse

I implement the sparse feature layer in [sparse.py](./sparse.py), which is a quite simple and clean implementation.


To run the mnist sparse training example

```
pip3 install -r requirements.txt
bash ./run_mnist_sparse.sh
```



To run the cifar10 training example (training on cpu is very slow, I train them using free gpu from [google colab](https://research.google.com/colaboratory/))

```bash
python3 ./cifar/main.py  # baseline, non-sparse

bash ./run_cifar_sparse.sh # sparse training, overall sparsity 50%
```

The cifar10 training result is pretty impressive as seen in logs [baseline.log](records/cifar.baseline.log) and [sparse.log](records/cifar.sparse.log). 


| cifar10 | baseline | sparse (50% for every convolution) |
|---------|----------|------------------------------------|
| acc     | 94.33%   | 93.98%                             |
