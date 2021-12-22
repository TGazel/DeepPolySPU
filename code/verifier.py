import argparse

import torch

from deeppoly import DP_Net, DP_Shape, DP_Verifier, DP_Loss
from networks import FullyConnected

DEVICE = "cpu"
INPUT_SIZE = 28
NUM_EPOCHS = 1000
OPTIM_LR = 0.1

if "cuda" in DEVICE and hasattr(torch.cuda, 'FloatTensor'):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore


def analyze(net, inputs, eps, true_label):
    dp_net = DP_Net(net)
    dp_verif = DP_Verifier(dp_net.num_labels, true_label)
    dp_loss = DP_Loss()

    optimizer = torch.optim.Adam(dp_net.parameters(), lr=OPTIM_LR)
    for _ in range(NUM_EPOCHS):
        optimizer.zero_grad()

        shape = DP_Shape.from_eps(inputs, eps, clamp=(0.0, 1.0))

        netshape = dp_net(shape)
        verifshape = dp_verif(netshape)

        if (verifshape.lbounds > 0.0).all():
            return True

        loss = dp_loss(verifshape)
        loss.backward()
        optimizer.step()
    return False


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net',
                        type=str,
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net.endswith('fc1'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net.endswith('fc2'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc3'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net.endswith('fc4'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc5'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
