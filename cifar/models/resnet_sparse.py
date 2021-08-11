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
from quantize import quantize, quantize_sequential


class SparseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, quantize_kwargs={}, **kwargs):
        super(SparseBasicBlock, self).__init__()
        self.use_quantize = ('merge_bn_step' in quantize_kwargs) and quantize_kwargs['merge_bn_step'] > 0
        if self.use_quantize:
            print('use_quantize=True in BasicBlock')

        self.need_shortcut = stride != 1 or in_planes != self.expansion*planes
        self.conv1 = quantize_sequential(nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(planes), **quantize_kwargs)
        self.conv2 = quantize_sequential(nn.Conv2d(planes, planes, kernel_size=3,
                                                   stride=1, padding=1, bias=False),  nn.BatchNorm2d(planes), **quantize_kwargs, no_quantize_output=self.need_shortcut)
        name = kwargs.pop('name')

        self.sp_conv1 = SparseLayer(**kwargs, name=name+'_conv1')
        self.sp_conv2 = SparseLayer(**kwargs, name=name+'_conv2')

        self.shortcut = nn.Sequential()
        if self.need_shortcut:
            self.shortcut = nn.Sequential(
                quantize_sequential(nn.Conv2d(in_planes, self.expansion*planes,
                                              kernel_size=1, stride=stride, bias=False),
                                    nn.BatchNorm2d(self.expansion*planes), **quantize_kwargs, no_quantize_output=True),
                SparseLayer(**kwargs, name=name+'_shortcut')
            )

    def forward(self, x):
        out = F.relu(self.sp_conv1(self.conv1(x)))
        out = self.sp_conv2(self.conv2(out))
        out += self.shortcut(x)
        if self.need_shortcut:
            out = quantize(out, 8, 5)  # quantize outside
        out = F.relu(out)
        return out


class SparseResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, sparsity=0., start_steps=20000, prune_freq=2000, n_prunes=5, quantize_step=-1, **kwargs):
        super(SparseResNet, self).__init__()
        self.in_planes = 64
        self.use_quantize = quantize_step > 0
        quantize_kwargs = {'merge_bn_step': quantize_step}
        if self.use_quantize:
            print('use_quantize=True in first layer')
        sparse_kwargs = dict(sparsity=sparsity, start_steps=start_steps, prune_freq=prune_freq, n_prunes=n_prunes)

        self.conv1 = quantize_sequential(nn.Conv2d(3, 64, kernel_size=3,
                                                   stride=1, padding=1, bias=False), nn.BatchNorm2d(64), **quantize_kwargs)
        self.sp_conv1 = SparseLayer(**sparse_kwargs, name='sparse_conv1')

        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], stride=1, **sparse_kwargs, name="sparse_block1", quantize_kwargs=quantize_kwargs)
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2, **sparse_kwargs, name="sparse_block2", quantize_kwargs=quantize_kwargs)
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2, **sparse_kwargs, name="sparse_block3", quantize_kwargs=quantize_kwargs)
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2, **sparse_kwargs, name="sparse_block4", quantize_kwargs=quantize_kwargs)
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
        if self.use_quantize:
            x = quantize(x, 8, 5)
        out = F.relu(self.sp_conv1(self.conv1(x)))
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
