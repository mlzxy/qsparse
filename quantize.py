import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy


class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, bits=8, n=5, channel_index=1, int_compute=False):
        limit = 2.0 ** (bits - 1)
        tof = 2.0 ** -n
        toi = 2.0 ** n
        shape = [1 for _ in input.shape]
        if isinstance(n, torch.Tensor) and sum(n.shape) > 1:
            shape[channel_index] = -1
            tof, toi = tof.view(*shape), toi.view(*shape)
        ctx.save_for_backward(limit, tof)
        q = (input * toi).int()
        q.clamp_(-limit, limit - 1)
        if int_compute:
            if input.dtype == torch.int:
                return input
            else:
                return q
        else:
            return q.float() * tof

    @staticmethod
    def backward(ctx, grad_output):
        limit, tof = ctx.saved_tensors
        return grad_output.clamp_(-limit * tof, (limit - 1) * tof)


quantize = Quantize.apply


class QuantizeLayer(nn.Module):
    def __init__(self, bits=8, n=5):
        self.bits = bits
        self.n = n

    def forward(self, x):
        return quantize(x, self.bits, self.n)


def calculate_best_n(tensor, bits):
    MIN_VALID_N = 1
    MAX_VALID_N = 15
    err = float('inf')
    best_n = None
    tensor = tensor.view(-1)  # flatten
    for n in range(MIN_VALID_N, MAX_VALID_N):
        tensor_q = quantize(m, bits, n)
        err_ = torch.sum((tensor - tensor_q)**2).item()
        if err_ < err:
            best_n = n
            err = err_
    return best_n


def merge_bn_conv(conv, bn, test=False):
    bn.eval()  # don't forget this !

    w = conv.weight.detach()
    b = conv.bias.detach() if conv.bias is not None else 0

    mean = bn.running_mean.detach()
    var_sqrt = torch.sqrt(bn.running_var.detach().add(1e-5))
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    new_weight = w * (gamma / var_sqrt)[:, None, None, None]
    new_bias = (b - mean) * gamma / var_sqrt + beta

    if test:
        inputs = torch.randn(1, conv.in_channels, 32, 32)
        with_bn = bn(conv(inputs))
        conv.weight = nn.Parameter(new_weight)
        conv.bias = nn.Parameter(new_bias)
        merged_bn = conv(inputs)
        print("merge_bn_conv.test", torch.allclose(with_bn, merged_bn, atol=1e-5))

    return new_weight, new_bias


def merge_bn_deconv(deconv, bn, test=False):
    bn.eval()  # don't forget this !

    w = deconv.weight.detach()
    b = deconv.bias.detach() if deconv.bias is not None else 0

    mean = bn.running_mean.detach()
    var_sqrt = torch.sqrt(bn.running_var.detach().add(1e-5))
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    new_weight = w * (gamma / var_sqrt)[None, :, None, None]
    new_bias = (b - mean) * gamma / var_sqrt + beta

    if test:
        inputs = torch.randn(1, deconv.in_channels, 32, 32)
        with_bn = bn(deconv(inputs))
        deconv.weight = nn.Parameter(new_weight)
        deconv.bias = nn.Parameter(new_bias)
        merged_bn = deconv(inputs)
        print("merge_bn_deconv.test", torch.allclose(with_bn, merged_bn, atol=1e-5))

    return new_weight, new_bias


def merge_bn_linear(linear, bn, test=False):
    bn.eval()  # don't forget this !

    w = linear.weight.detach()
    b = linear.bias.detach() if linear.bias is not None else 0

    mean = bn.running_mean.detach()
    var_sqrt = torch.sqrt(bn.running_var.detach().add(1e-5))
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    new_weight = w * (gamma / var_sqrt)[:, None]
    new_bias = (b - mean) * gamma / var_sqrt + beta

    if test:
        inputs = torch.randn(1, linear.in_features)
        with_bn = bn(linear(inputs))
        linear.weight = nn.Parameter(new_weight)
        linear.bias = nn.Parameter(new_bias)
        merged_bn = linear(inputs)
        print("merge_bn_linear.test", torch.allclose(with_bn, merged_bn, atol=1e-5))

    return new_weight, new_bias


class BatchNormQuantizer(nn.Module):
    """
    the output of this layer is quantized
    """

    def __init__(
        self,
        # conv/deconv/fc layer
        op,

        # batch norm layer
        bn=None,

        # quantization parameters
        bits=8,
        nw=6,
        nb=6,
        no=5,
        quantize_weight_per_channel=True,
        merge_bn_step=10000,

        # debug
        name='',

        # integer computation for hardware tests
        int_compute=False,
        ni=-1
    ):
        super(BatchNormQuantizer, self).__init__()
        # status
        self._init = False
        self._bn_merged = False
        self._n_updates = 0

        # internal layer
        self.op = [op, ]  # wrap it with a list to prevent double reference to the same weight
        self.weight = op.weight
        self.bias = op.bias

        if isinstance(op, nn.Conv2d):
            self.F = lambda inp, weight, bias: F.conv2d(
                inp, weight, bias=bias, stride=op.stride, padding=op.padding, dilation=op.dilation, groups=op.groups)
            self.merge_bn = merge_bn_conv
            self.channel_index = 0
        elif isinstance(op, nn.ConvTranspose2d):
            def func(inp, weight, bias):
                if self.int_compute:
                    inp = inp.float()
                    weight = weight.float()
                    bias = bias.float() if bias is not None else bias
                r = F.conv_transpose2d(
                    inp, weight, bias=bias, stride=op.stride, padding=op.padding, dilation=op.dilation, groups=op.groups)
                if self.int_compute:
                    r = r.int()
                return r

            self.F = func
            self.merge_bn = merge_bn_deconv
            self.channel_index = 1
        elif isinstance(op, nn.Linear):
            self.F = F.linear
            self.merge_bn = merge_bn_linear
            self.channel_index = 0
        else:
            raise RuntimeError("Unknown operation type")

        self.bn = bn

        # parameters
        self.name = name
        self.merge_bn_step = merge_bn_step
        self._no = no
        self._nw = nw
        self._nb = nb
        self._ni = ni
        self._bits = bits
        self.quantize_weight_per_channel = quantize_weight_per_channel
        self.int_compute = int_compute

    def int(self, flag=True):
        self.int_compute = flag
        return self

    def forward(self, x):
        if self.int_compute:
            assert x.dtype == torch.int
            assert self._ni > 0, "input n must be given"

        if not self._init:
            self.bits = nn.Parameter(
                torch.tensor(self._bits, dtype=torch.int, requires_grad=False, device=x.device),
                requires_grad=False,
            )
            self.no = nn.Parameter(
                torch.tensor(self._no, dtype=torch.int, requires_grad=False, device=x.device),
                requires_grad=False,
            )
            self.nw = nn.Parameter(
                torch.tensor(([self._nw, ] * self.weight.shape[self.channel_index]) if self.quantize_weight_per_channel else self._nw,
                             dtype=torch.int, requires_grad=False, device=x.device),
                requires_grad=False,
            )
            self.nb = nn.Parameter(
                torch.tensor(([self._nb, ] * self.weight.shape[self.channel_index]) if self.quantize_weight_per_channel else self._nb,
                             dtype=torch.int, requires_grad=False, device=x.device),
                requires_grad=False,
            )
            self._init = True

        if self._n_updates >= self.merge_bn_step:
            if not self._bn_merged:
                with torch.no_grad():
                    print(f'[Quantizer] @ {self._n_updates} # {self.name} -> Merge BN')
                    self.op[0].weight = self.weight
                    self.op[0].bias = self.bias
                    weight, bias = self.merge_bn(self.op[0], self.bn)
                    self.weight.data = weight
                    self.bias = nn.Parameter(bias, requires_grad=True)
                    if self.quantize_weight_per_channel:
                        print(f'[Quantizer] @ {self._n_updates} # {self.name} -> Quantize weights by MSE')
                        for i in range(weight.shape[0]):
                            bits = self.bits.item()
                            ws = calculate_best_n(weight[i] if self.channel_index == 0 else weight[:, i], bits)
                            bs = calculate_best_n(bias[i], bits)
                            self.nw[i] = ws
                            self.nb[i] = bs
                    self._bn_merged = True

        n_m = self.nw + self._ni
        weight = quantize(self.weight, self.bits, self.nw, self.channel_index, self.int_compute)
        if self.bias is not None:
            bias = quantize(self.bias, self.bits, self.nb, 0, False)
            if self.int_compute:
                # since this happens in temporary buffer, so bits doesn't matter
                bias = quantize(bias, torch.tensor(31), n_m, 0, self.int_compute)
        else:
            bias = None

        out = self.F(x, weight, bias)
        if not self.int_compute:
            if (not self._bn_merged) and (self.bn is not None):
                out = self.bn(out)
        out = quantize(out, self.bits, self.no, 0, self.int_compute)

        if self.int_compute:
            nw = self.nw.detach()
            n_diff = n_m - self._no
            if self.quantize_weight_per_channel:
                for i in range(out.shape[1]):
                    out[:, i] = (out[:, i].float() / (2 ** n_diff[i].item())).int()
            else:
                out = out >> n_diff.item()

        if self.training:
            self._n_updates += 1
        return out


if __name__ == "__main__":
    # verify the quantization match between int and float domain, 1 hour today
    x = torch.rand(10)
    print('========= quantization test (should be quite close) ==========')
    print(x)
    print(quantize(x, 8, 7))
    uqx = (quantize(x, 8, 7) * (2**7)).int().float() / (2**7)
    uqx2 = quantize(x, 8, 7, 0, True).float() / (2**7)
    print('should be very closed to 0', torch.sum((x - uqx) ** 2), torch.sum((x - uqx2) ** 2))
    uqx3 = quantize(uqx2, 8, 7, 0, True).float() / (2**7)
    print('should be 0', torch.sum((uqx2 - uqx3) ** 2))
    print('\n')

    print('========= merge bn conv / deconv / fc tests ==========')
    bn = nn.BatchNorm2d(3)
    bn.weight.data = torch.rand(*bn.weight.shape)
    bn.bias.data = torch.rand(*bn.bias.shape)
    conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2)
    merge_bn_conv(conv, bn, test=True)

    bn = nn.BatchNorm1d(20)
    bn.weight.data = torch.rand(*bn.weight.shape)
    bn.bias.data = torch.rand(*bn.bias.shape)
    linear = nn.Linear(10, 20)
    merge_bn_linear(linear, bn, test=True)

    bn = nn.BatchNorm2d(3)
    bn.weight.data = torch.rand(*bn.weight.shape)
    bn.bias.data = torch.rand(*bn.bias.shape)
    deconv = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=2)
    merge_bn_deconv(deconv, bn, test=True)

    print('\n')

    print('========= quantized conv / deconv / fc tests ==========')
    inp_conv = torch.rand(1, 10, 32, 32)
    inp_fc = torch.rand(1, 10)

    fc = nn.Linear(10, 30)
    conv = nn.Conv2d(10, 20, 3, stride=1, padding=1)
    deconv = nn.ConvTranspose2d(10, 20, 3, stride=2, padding=1)

    with torch.no_grad():
        layers = [(inp_fc, fc), (inp_conv, conv), (inp_conv, deconv)]
        for inp, op in layers:
            n_inp = 6
            n_out = 5
            inp = quantize(inp, 8, n_inp)
            inpi = quantize(inp, 8, n_inp, 0, True)
            mod = BatchNormQuantizer(op, no=n_out, ni=n_inp)
            out_f = mod(inp)
            mod.int()
            out_i = mod(inpi)
            print('should be 0', torch.sum((out_i - out_f * (2 ** n_out)) ** 2))

    print('\n')
