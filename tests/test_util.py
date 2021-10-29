import torch

from qsparse import auto_name_prune_quantize_layers, prune, quantize


def test_auto_name_prune_quantize_layers():
    class TwoLayerNet(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            super(TwoLayerNet, self).__init__()
            self.linear1 = quantize(prune(torch.nn.Linear(D_in, H)))
            self.linear2 = quantize(prune(torch.nn.Linear(H, D_out)))

        def forward(self, x):
            h_relu = self.linear1(x).clamp(min=0)
            y_pred = self.linear2(h_relu)
            return y_pred

    net = TwoLayerNet(10, 30, 1)
    net = auto_name_prune_quantize_layers(net)
    assert net.linear1.prune.name == "linear1.prune"
    assert net.linear1.quantize.name == "linear1.quantize"
    assert net.linear2.prune.name == "linear2.prune"
    assert net.linear2.quantize.name == "linear2.quantize"
