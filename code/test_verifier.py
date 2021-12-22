import torch
import torch.nn as nn

from verifier import analyze


class ExampleNet(nn.Sequential):
    def __init__(self):
        layer_0 = nn.Linear(2, 2)
        layer_0.weight = nn.Parameter(torch.Tensor([[1, 1], [1, -1]]))
        layer_0.bias = torch.nn.Parameter(torch.zeros(1, 2))

        layer_1 = nn.ReLU()

        layer_2 = nn.Linear(2, 2)
        layer_2.weight = nn.Parameter(torch.Tensor([[1, 1], [1, 1]]))
        layer_2.bias = torch.nn.Parameter(torch.Tensor([[0.5, 0]]))

        layer_3 = nn.ReLU()

        layer_4 = nn.Linear(2, 2)
        layer_4.weight = nn.Parameter(torch.Tensor([[-1, 1], [0, 1]]))
        layer_4.bias = torch.nn.Parameter(torch.Tensor([[3, 0]]))

        super().__init__(
            layer_0,
            layer_1,
            layer_2,
            layer_3,
            layer_4
        )


if __name__ == '__main__':
    net = ExampleNet()
    inputs = torch.FloatTensor([[0., 0.]])

    if analyze(net, inputs, eps=1.0, true_label=0):
        print('verified')
    else:
        print('not verified')
