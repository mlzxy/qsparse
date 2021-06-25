'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from sparse import SparseLayer


class SparseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        name = kwargs.pop('name')

        self.sp_conv1 = SparseLayer(**kwargs,name=name+'_conv1')
        self.sp_conv2 = SparseLayer(**kwargs,name=name+'_conv2')

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                SparseLayer(**kwargs, name=name+'_shortcut')
            )

    def forward(self, x):
        out = F.relu(self.sp_conv1(self.bn1(self.conv1(x))))
        out = self.sp_conv2(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SparseBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(SparseBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.sp_conv1 = SparseLayer(**kwargs,name=name+'_conv2')
        self.sp_conv2 = SparseLayer(**kwargs,name=name+'_conv2')
        self.sp_conv3 = SparseLayer(**kwargs,name=name+'_conv3')

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                SparseLayer(**kwargs, name=name+'_shortcut')
            )

    def forward(self, x):
        out = F.relu(self.sp_conv1(self.bn1(self.conv1(x))))
        out = F.relu(self.sp_conv1(self.bn2(self.conv2(out))))
        out = self.sp_conv3(self.bn3(self.conv3(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SparseResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, sparsity=0., start_steps=20000, prune_freq=2000, n_prunes=5):
        super(SparseResNet, self).__init__()
        self.in_planes = 64

        sparse_kwargs = dict(sparsity=sparsity, start_steps=start_steps, prune_freq=prune_freq, n_prunes=n_prunes)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.sp_conv1 = SparseLayer(**sparse_kwargs, name='sparse_conv1')

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, **sparse_kwargs, name="sparse_block1")
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, **sparse_kwargs, name="sparse_block2")
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, **sparse_kwargs, name="sparse_block3")
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, **sparse_kwargs, name="sparse_block4")
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        name = kwargs['name']
        for i, stride in enumerate(strides):
            kwargs['name'] = name + f'_{i}'
            layers.append(block(self.in_planes, planes, stride, **kwargs))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.sp_conv1(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SparseResNet18(**kwargs):
    return SparseResNet(SparseBasicBlock, [2, 2, 2, 2], **kwargs)


def SparseResNet34(**kwargs):
    return SparseResNet(SparseBasicBlock, [3, 4, 6, 3], **kwargs)


def SparseResNet50(**kwargs):
    return SparseResNet(SparseBottleneck, [3, 4, 6, 3], **kwargs)


def SparseResNet101(**kwargs):
    return SparseResNet(SparseBottleneck, [3, 4, 23, 3], **kwargs)


def SparseResNet152(**kwargs):
    return SparseResNet(SparseBottleneck, [3, 8, 36, 3], **kwargs)


def test():
    net = SparseResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
