import csv
import pathlib
import sys
import unittest

import torch

from networks import FullyConnected
from verifier import analyze, INPUT_SIZE, DEVICE
from deeppoly import DP_Shape, DP_Net, DP_Verifier

VERBOSE = False


class TestVerifier(unittest.TestCase):
    def load_spec(self, gt_path, net_name, spec):
        spec_path = gt_path.parent / net_name / spec
        self.assertTrue(spec_path.is_file())

        with open(spec_path, "r") as f:
            lines = [line[:-1] for line in f.readlines()]
            true_label = int(lines[0])
            pixel_values = [float(line) for line in lines[1:]]
            eps = float(spec_path.stem.split("_")[-1])

        return true_label, pixel_values, eps

    def load_net(self, gt_path, net_name):
        if net_name.endswith("fc1"):
            layers = [50, 10]
        elif net_name.endswith("fc2"):
            layers = [100, 50, 10]
        elif net_name.endswith("fc3"):
            layers = [100, 100, 10]
        elif net_name.endswith("fc4"):
            layers = [100, 100, 50, 10]
        elif net_name.endswith("fc5"):
            layers = [100, 100, 100, 100, 10]
        else:
            self.fail(f"Unknown net {net_name}")

        net = FullyConnected(DEVICE, INPUT_SIZE, layers).to(DEVICE)

        net_path = gt_path.parent.parent / "mnist_nets" / f"{net_name}.pt"
        net.load_state_dict(torch.load(net_path, map_location=torch.device(DEVICE)))

        return net

    def debug_lbounds(self, net, inputs, eps, true_label):
        dp_net = DP_Net(net)
        dp_verif = DP_Verifier(dp_net.num_labels, true_label)
        shape = DP_Shape.from_eps(inputs, eps, clamp=(0.0, 1.0))
        netshape = dp_net(shape)
        verifshape = dp_verif(netshape)
        return verifshape.lbounds

    def subtest_verifier(self, net, pixel_values, true_label, eps, gt):
        inputs = (
            torch.FloatTensor(pixel_values)
            .view(1, 1, INPUT_SIZE, INPUT_SIZE)
            .to(DEVICE)
        )

        outs = net(inputs)
        pred_label = outs.max(dim=1)[1].item()
        self.assertEqual(pred_label, true_label)

        verified = analyze(net, inputs, eps, true_label)
        lbounds = self.debug_lbounds(net, inputs, eps, true_label)

        if gt == "verified":
            self.assertTrue(
                verified,
                msg=f'Actual "not verified", expected "verified" (0 pt)\n'
                f"lbounds={lbounds}",
            )
        elif gt == "not verified":
            self.assertFalse(
                verified,
                msg=f'Actual "verified", expected "not verified" (-2 pt)\n'
                f"lbounds={lbounds}",
            )
        else:
            self.fail()

    def test_verifier(self):
        if VERBOSE:
            print()

        gt_path = (
            pathlib.Path(__file__).absolute()
            .parent
            .parent
            / "test_cases"
            / "gt.txt"
        )

        self.assertTrue(gt_path.is_file())

        with open(gt_path, "r") as gt_file:
            gt_reader = csv.reader(gt_file)

            for (net_name, spec, gt) in gt_reader:
                true_label, pixel_values, eps = self.load_spec(gt_path, net_name, spec)

                net = self.load_net(gt_path, net_name)
                with self.subTest(f"{net_name} {spec}"):
                    if VERBOSE:
                        print(f"-- {net_name} {spec}")
                    self.subtest_verifier(net, pixel_values, true_label, eps, gt)

    def test_verifier_prelim(self):
        if VERBOSE:
            print()

        gt_path = (
            pathlib.Path(__file__).absolute()
            .parent
            .parent
            / "prelim_test_cases"
            / "gt.txt"
        )

        self.assertTrue(gt_path.is_file())

        with open(gt_path, "r") as gt_file:
            gt_reader = csv.reader(gt_file)

            for (net_name, spec, gt) in gt_reader:
                true_label, pixel_values, eps = self.load_spec(gt_path, net_name, spec)

                net = self.load_net(gt_path, net_name)
                with self.subTest(f"{net_name} {spec}"):
                    if VERBOSE:
                        print(f"-- {net_name} {spec}")
                    self.subtest_verifier(net, pixel_values, true_label, eps, gt)


if __name__ == "__main__":
    if "-v" in sys.argv or "--verbose" in sys.argv:
        VERBOSE = True

    unittest.main()
