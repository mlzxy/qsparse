# pytorch-feature-sparse

I implement the sparse feature layer in [sparse.py](./sparse.py), which is a quite simple and clean implementation.


**MNIST**

To run the mnist sparse training example

```
pip3 install -r requirements.txt
bash ./run_mnist_sparse.sh
```

**CIFAR10**

To run the cifar10 training example (training on cpu is very slow, I train them using free gpu from [google colab](https://research.google.com/colaboratory/))

```bash
python3 ./cifar/main.py  # baseline, non-sparse

bash ./run_cifar_sparse.sh # sparse training, overall sparsity 50%
```

The cifar10 training result is pretty impressive as seen in logs [baseline.log](records/cifar.baseline.log) and [sparse.log](records/cifar.sparse.log). 


| cifar10 | baseline | sparse (50% for every convolution) |
|---------|----------|------------------------------------|
| acc     | 94.33%   | 94.39%                             |



**Pix2Pix (Image To Image Translation Task)** 

To run the pix2pix training

```bash
cd ./cyclegan_pix2pix
bash ./run_pix2pix_baseline.sh  # baseline from the original repo https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
bash ./run_pix2pix_sparse.sh  # 50% sparse for every convolution & deconvolution, except for the final layer 
```

Some demo images generated from [baseline](records/pix2pix/baseline/images) and [sparse network (50%)](records/pix2pix/sparse/images). Comparing to the classification case, which is relatively easier, we have made some small tweaks to make this case works. In fact, we could say the image quality is very similar visually 😃, even though the sparse network generally has a higher training loss. 