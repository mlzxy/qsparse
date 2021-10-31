from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
import torch.optim as optim
import sys

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from qsparse import prune, quantize
from qsparse.util import auto_name_prune_quantize_layers
from enum import Enum


def identity(x):
    return x


def create_p_q(train_mode, epoch_size):
    def bypass(*args):
        if len(args) == 0:
            return identity
        else:
            return args[0]

    quantize_first = train_mode.startswith("quantize")
    if "bs" in train_mode:
        bs = int(train_mode[train_mode.find("bs") + 2 :])
    else:
        bs = 8

    def q(*args, c=0):
        if "quantize" in train_mode:
            return (
                quantize(
                    timeout=epoch_size * (3 if quantize_first else 7),
                    channelwise=-1,
                    decimal_range=(1, 7),
                    window_size=bs,
                )
                if len(args) == 0
                else quantize(
                    args[0],
                    timeout=epoch_size * (2 if quantize_first else 6),
                    channelwise=c or 1,
                    bias_bits=20,
                )
            )
        else:
            return bypass(*args)

    def p(*args):
        kw = {
            "start": epoch_size * (4 if quantize_first else 2),
            "interval": epoch_size * 1,
            "repetition": 3,
            "sparsity": 0.5,
        }
        if "weight" in train_mode:
            return identity if len(args) == 0 else prune(args[0], **kw)
        elif "feat" in train_mode:
            return prune(**kw, window_size=bs) if len(args) == 0 else args[0]
        else:
            return bypass(*args)

    return p, q


class Net(nn.Module):
    def __init__(self, train_mode="float", epoch_size=-1):
        super(Net, self).__init__()
        p, q = create_p_q(train_mode, epoch_size)

        self.qin = q()

        self.conv1 = q(nn.Conv2d(1, 32, 3, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.p1, self.q1 = p(), q()

        self.conv2 = q(p(nn.Conv2d(32, 64, 3, 1)))
        self.bn2 = nn.BatchNorm2d(64)
        self.p2, self.q2 = p(), q()

        self.fc1 = q(p(nn.Linear(9216, 128)), c=-1)
        self.bn3 = nn.BatchNorm1d(128)
        self.p3, self.q3 = p(), q()

        self.fc2 = q(nn.Linear(128, 10), c=-1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.qin(x)
        x = F.relu(self.q1(self.p1(self.bn1(self.conv1(x)))))
        x = F.relu(self.q2(self.p2(self.bn2(self.conv2(x)))))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.q3(self.p3(self.bn3(self.fc1(x)))))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("./data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("./data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    print(f"training epoch size = {len(train_loader)}")
    model = Net(epoch_size=len(train_loader)).to(device)
    auto_name_prune_quantize_layers(model)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
