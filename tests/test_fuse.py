import torch
import torch.nn as nn

from qsparse import fuse_bn


def test_conv2d():
    net = nn.Sequential(nn.Conv2d(3, 10, 3), nn.BatchNorm2d(10))
    net.train()
    inputs = torch.randn(10, 3, 32, 32)
    nn.init.uniform_(list(net.children())[1].weight)
    nn.init.uniform_(list(net.children())[1].bias)
    net(inputs)
    net.eval()
    gt = net(inputs)
    net_fused = fuse_bn(net)
    pred = net_fused(inputs)
    assert "batchnorm" not in str(net_fused).lower()
    assert torch.allclose(gt, pred, atol=1e-5)


def test_linear():
    net = nn.Sequential(nn.Linear(20, 10), nn.BatchNorm1d(10))
    net.train()
    inputs = torch.randn(10, 20)
    nn.init.uniform_(list(net.children())[1].weight)
    nn.init.uniform_(list(net.children())[1].bias)
    net(inputs)
    net.eval()
    gt = net(inputs)
    net_fused = fuse_bn(net)
    pred = net_fused(inputs)
    assert "batchnorm" not in str(net_fused).lower()
    assert torch.allclose(gt, pred, atol=1e-5)


def test_deconv2d():
    net = nn.Sequential(nn.ConvTranspose2d(3, 10, 3), nn.BatchNorm2d(10))
    net.train()
    inputs = torch.randn(10, 3, 32, 32)
    nn.init.uniform_(list(net.children())[1].weight)
    nn.init.uniform_(list(net.children())[1].bias)
    net(inputs)
    net.eval()
    gt = net(inputs)
    net_fused = fuse_bn(net)
    pred = net_fused(inputs)
    assert "batchnorm" not in str(net_fused).lower()
    assert torch.allclose(gt, pred, atol=1e-5)


def test_complex_sequential():
    inputs = torch.randn(10, 10, 32, 32)
    net1 = nn.Sequential(nn.Conv2d(10, 10, 3), nn.BatchNorm2d(10))
    net1.train()
    nn.init.uniform_(list(net1.children())[1].weight)
    nn.init.uniform_(list(net1.children())[1].bias)
    net1(inputs)

    net2 = nn.Sequential(nn.ConvTranspose2d(10, 10, 3), nn.BatchNorm2d(10))
    net2.train()
    nn.init.uniform_(list(net2.children())[1].weight)
    nn.init.uniform_(list(net2.children())[1].bias)
    net2(inputs)

    net3 = nn.Sequential(nn.Conv2d(10, 10, 3), nn.BatchNorm2d(10))
    net3.train()
    nn.init.uniform_(list(net3.children())[1].weight)
    nn.init.uniform_(list(net3.children())[1].bias)
    net3(inputs)

    net4 = nn.Sequential(nn.ConvTranspose2d(10, 10, 3), nn.BatchNorm2d(10))
    net4.train()
    nn.init.uniform_(list(net4.children())[1].weight)
    nn.init.uniform_(list(net4.children())[1].bias)
    net4(inputs)

    bn = nn.BatchNorm2d(10)
    nn.init.uniform_(bn.weight)
    nn.init.uniform_(bn.bias)
    bn(inputs)

    net = nn.Sequential(
        net1, nn.Sequential(net3, net4), net2, nn.Conv2d(10, 10, 3), nn.Sequential(bn)
    )
    net.eval()
    gt = net(inputs)
    net_fused = fuse_bn(net, inplace=False)
    assert "batchnorm" not in str(net_fused).lower()
    pred = net_fused(inputs)
    assert torch.allclose(gt, pred, atol=1e-5)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                net1,
                nn.Sequential(net3, net4),
                net2,
                nn.Conv2d(10, 10, 3),
                nn.Sequential(bn),
            )

        def forward(self, x):
            return self.net(x)

    net = Net()
    net.eval()
    gt = net(inputs)
    net_fused = fuse_bn(net, inplace=False)
    assert "batchnorm" not in str(net_fused).lower()
    pred = net_fused(inputs)
    assert torch.allclose(gt, pred, atol=1e-5)


def test_data_parallel():
    net = nn.Sequential(nn.Linear(20, 10), nn.BatchNorm1d(10))
    net.train()
    inputs = torch.randn(10, 20)
    nn.init.uniform_(list(net.children())[1].weight)
    nn.init.uniform_(list(net.children())[1].bias)

    net = nn.DataParallel(net)
    net(inputs)
    net.eval()
    gt = net(inputs)
    net_fused = fuse_bn(net)
    pred = net_fused(inputs)
    assert "batchnorm" not in str(net_fused).lower()
    assert torch.allclose(gt, pred, atol=1e-5)
