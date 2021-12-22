from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import networks

# Custom Python types
ClampType = Tuple[Union[float, None], Union[float, None]]

TransformType = Optional[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]


class DP_Shape:
    def __init__(self, lbounds: torch.Tensor, ubounds: torch.Tensor) -> None:
        self.parent: Optional[DP_Shape] = None
        self._lbounds: Optional[torch.Tensor] = lbounds
        self._ubounds: Optional[torch.Tensor] = ubounds
        self.transform: TransformType = None

        assert lbounds.shape == ubounds.shape and (lbounds <= ubounds).all(), \
            "Invalid bounding boxes."

    @classmethod
    def from_transform(cls,
                       parent: 'DP_Shape',
                       ineql: torch.Tensor,
                       biasl: torch.Tensor,
                       inequ: torch.Tensor,
                       biasu: torch.Tensor
                       ) -> 'DP_Shape':
        self = cls.__new__(cls)
        self.parent = parent
        self._lbounds = None
        self._ubounds = None
        self.transform = (ineql, biasl, inequ, biasu)
        return self

    @classmethod
    def from_eps(
        cls,
        inputs: torch.Tensor,
        eps: torch.Tensor,
        clamp: ClampType
    ) -> 'DP_Shape':
        return cls((inputs - eps).clamp(*clamp), (inputs + eps).clamp(*clamp))

    @property
    def lbounds(self) -> torch.Tensor:
        assert self._lbounds is not None, \
            "You must call backsub(...) on a transformed DP_Shape before " \
            "accessing its bounds."
        return self._lbounds

    @property
    def ubounds(self) -> torch.Tensor:
        assert self._ubounds is not None, \
            "You must call backsub(...) on a transformed DP_Shape before " \
            "accessing its bounds."
        return self._ubounds

    def __repr__(self):
        if self._lbounds is None or self._ubounds is None:
            return f'{self.__class__.__name__}(...)'

        return f'{self.__class__.__name__}(\n' + \
               f'    lbounds={self.lbounds},\n' + \
               f'    ubounds={self.ubounds}\n' + \
               ')'

    def resolve(self,
                ineql: torch.Tensor,
                biasl: torch.Tensor,
                inequ: torch.Tensor,
                biasu: torch.Tensor,
                root: 'DP_Shape'):
        Wl, bl, Wu, bu = ineql, biasl, inequ, biasu

        # HACK: Unless teaching staff lets us convert everything to double, we
        # have to create a FloatTensor containing zero (otherwise we have a
        # RuntimeError when using Python's double '0.')
        zero = torch.zeros(1)

        Wl_pos = torch.where(Wl > zero, Wl, zero)
        Wl_neg = torch.where(Wl < zero, Wl, zero)

        Wu_pos = torch.where(Wu > zero, Wu, zero)
        Wu_neg = torch.where(Wu < zero, Wu, zero)

        self._lbounds = root.lbounds @ Wl_pos.T + root.ubounds @ Wl_neg.T + bl
        self._ubounds = root.ubounds @ Wu_pos.T + root.lbounds @ Wu_neg.T + bu

    def backsub(self):
        if self._lbounds is not None and self._ubounds is not None:
            return

        shape = self
        fineql, fbiasl, finequ, fbiasu = self.transform

        shape = shape.parent
        while shape.transform is not None:
            sineql, sbiasl, sinequ, sbiasu = shape.transform

            # HACK: Unless teaching staff lets us convert everything to double,
            # we have to create a FloatTensor containing zero (otherwise we
            # have a RuntimeError when using Python's double '0.')
            zero = torch.zeros(1)

            fineql_pos = torch.where(fineql > zero, fineql, zero)
            fineql_neg = torch.where(fineql < zero, fineql, zero)

            finequ_pos = torch.where(finequ > zero, finequ, zero)
            finequ_neg = torch.where(finequ < zero, finequ, zero)

            fbiasl = sbiasl @ fineql_pos.T + sbiasu @ fineql_neg.T + fbiasl
            fbiasu = sbiasu @ finequ_pos.T + sbiasl @ finequ_neg.T + fbiasu

            fineql = fineql_pos @ sineql + fineql_neg @ sinequ
            finequ = finequ_pos @ sinequ + finequ_neg @ sineql

            shape = shape.parent

        self.resolve(fineql, fbiasl, finequ, fbiasu, shape)


class DP_Linear(nn.Module):
    def __init__(self, inner: nn.Linear):
        super().__init__()
        self.in_features = inner.in_features
        self.out_features = inner.out_features

        self.weight = inner.weight.detach()
        self.bias = inner.bias.detach()

    def forward(self, in_shape: DP_Shape) -> DP_Shape:
        W, b = self.weight, self.bias

        # Encode exact affine transform
        ineql, biasl = W, b
        inequ, biasu = W, b

        return DP_Shape.from_transform(in_shape, ineql, biasl, inequ, biasu)


class DP_ReLU(nn.Module):
    def __init__(self, _inner: nn.ReLU):
        super().__init__()

    def forward(self, in_shape: DP_Shape) -> DP_Shape:
        in_shape.backsub()

        lbounds, ubounds = in_shape.lbounds, in_shape.ubounds
        _, N = lbounds.shape

        assert lbounds.shape == ubounds.shape == (1, N), \
            f"lbounds.shape={lbounds.shape}, ubounds.shape={ubounds.shape}"

        # Affine function defining the hypotenuse.
        slope = (ubounds / (ubounds - lbounds)).flatten()
        slope[slope != slope] = 0.  # Replace NaNs from zero-div with zero.
        intercept = (1 - slope) * ubounds

        # Case 1: Below-zero, destroy.
        below = (ubounds < 0.).flatten()
        ineql_below, biasl_below = torch.zeros(N), torch.zeros(1, N)
        inequ_below, biasu_below = torch.zeros(N), torch.zeros(1, N)

        # Case 2: Above-zero, keep.
        above = (lbounds > 0.).flatten()
        ineql_above, biasl_above = \
            torch.where(above, torch.ones(N), ineql_below), biasl_below
        inequ_above, biasu_above = \
            torch.where(above, torch.ones(N), inequ_below), biasu_below

        # Case 3: Crossing, encode triangular shape.
        crossing = ~below & ~above
        ineql, biasl = ineql_above, biasl_above
        inequ = torch.where(crossing, slope, inequ_above)
        biasu = torch.where(crossing, intercept, biasu_above)

        ineql = torch.diag(ineql)
        inequ = torch.diag(inequ)

        return DP_Shape.from_transform(in_shape, ineql, biasl, inequ, biasu)


class DP_Flatten(nn.Module):
    def __init__(self, inner: nn.Flatten):
        super().__init__()
        self.flattener = inner

    def forward(self, in_shape: DP_Shape) -> DP_Shape:
        lbounds = self.flattener(in_shape.lbounds)
        ubounds = self.flattener(in_shape.ubounds)

        return DP_Shape(lbounds, ubounds)


class DP_Normalization(nn.Module):
    def __init__(self, inner: networks.Normalization) -> None:
        super().__init__()
        # `mean` and `sigma` are initialized in networks.Normalization
        self.mean = inner.mean
        self.sigma = inner.sigma

    def forward(self, in_shape: DP_Shape) -> DP_Shape:
        lbounds = (in_shape.lbounds - self.mean) / self.sigma
        ubounds = (in_shape.ubounds - self.mean) / self.sigma

        return DP_Shape(lbounds, ubounds)


class DP_SPU(nn.Module):
    def __init__(self, inner: networks.SPU, previous: Optional[nn.Module]):
        super().__init__()
        self.spu = inner

        if previous is None:
            N = 1
        elif isinstance(previous, nn.Linear):
            N = previous.out_features
        else:
            class_name = previous.__class__.__name__
            raise AssertionError(
                f'Unsupported previous layer before SPU: {class_name}'
            )

        assert N >= 1, f"Invalid previous.out_features = {N}"

        mu, sigma = torch.zeros(N), torch.ones(N)
        self.p_uul0 = torch.nn.Parameter(torch.normal(mu, sigma))
        self.p_llg0 = torch.nn.Parameter(torch.normal(mu, sigma))
        self.p_cb = torch.nn.Parameter(torch.normal(mu, sigma))

    def compute_bounds(self, in_shape: DP_Shape):
        lbounds, ubounds = in_shape.lbounds, in_shape.ubounds
        _, N = lbounds.shape
        spu = self.spu

        # Functions for computing slopes of tangents and intercepts:
        def spu_slope(x):
            sigmoid_prime = -torch.sigmoid(-x) * (1.0 - torch.sigmoid(-x))
            return torch.where(x < 0, sigmoid_prime, 2 * x).flatten()

        def spu_intercept(x):
            return spu(x) - spu_slope(x) * x

        # FIXME: Potentially, replace 1e-6 with a zero-check instead.
        lu_slope = (
            (spu(ubounds) - spu(lbounds)) / (ubounds - lbounds + 1e-6)
        ).flatten()
        lu_intercept = spu(ubounds) - lu_slope * ubounds

        # Basis box case for soundness 
        inequ, biasu = torch.zeros(N), -0.5 * torch.ones(1, N)
        ineql = torch.zeros(N)
        biasl = torch.max(spu(lbounds), spu(ubounds)).reshape(1, N)

        below = (ubounds < 0.).flatten()
        pubelow = torch.sigmoid(self.p_uul0)*(ubounds-lbounds) + lbounds
        inequ, biasu = \
            torch.where(below, spu_slope(pubelow), inequ), \
            torch.where(below, spu_intercept(pubelow), biasu)
        ineql, biasl = \
            torch.where(below, lu_slope, ineql), \
            torch.where(below, lu_intercept, biasl)

        above = (lbounds > 0.).flatten()
        plabove = torch.sigmoid(self.p_llg0) * (ubounds - lbounds) + lbounds

        inequ, biasu = \
            torch.where(above, lu_slope, inequ), \
            torch.where(above, lu_intercept, biasu)
        ineql, biasl = \
            torch.where(above, spu_slope(plabove), ineql), \
            torch.where(above, spu_intercept(plabove), biasl)

        crossing = ~below & ~above

        slopecb = torch.sigmoid(self.p_cb) * (
            torch.max(
                spu_slope(
                    torch.sigmoid(self.p_uul0) * (ubounds - lbounds) + lbounds
                ),
                torch.zeros(1, N)
            )
            - (spu(lbounds) + 0.5) / (lbounds + 1e-6)
        ) + (spu(lbounds) + 0.5) / (lbounds + 1e-6)

        slopecb = slopecb.flatten()

        inequ, biasu = \
            torch.where(crossing, lu_slope, inequ), \
            torch.where(crossing, lu_intercept, biasu)
        ineql = torch.where(crossing, slopecb, ineql)
        biasl = torch.where(
            crossing,
            torch.where(
                slopecb < 0,
                -0.5 * torch.ones(1, N),
                -(slopecb**2) / 4 - 0.5 * torch.ones(1, N)
            ),
            biasl
        )

        # Special care for soundness in the case that u < (0.5 ** 0.5).
        # TODO: Adapt upper bound, maybe linked with the below case.
        # Maybe aim for a smoother transition when u tends towards 0,
        # carefully taking tangent passing through u at p in (l, pubelow).
        puvalley = lbounds
        valley = (spu_slope(puvalley) > lu_slope).flatten()
        crossingvalley = crossing & valley
        inequ, biasu = \
            torch.where(crossingvalley, spu_slope(puvalley[0]), inequ), \
            torch.where(crossingvalley, spu_intercept(puvalley[0]), biasu)

        return ineql, biasl, inequ, biasu, (below, above, valley)

    def forward(self, in_shape: DP_Shape) -> DP_Shape:
        in_shape.backsub()
        ineql, biasl, inequ, biasu, _ = self.compute_bounds(in_shape)

        ineql = torch.diag(ineql)
        inequ = torch.diag(inequ)

        return DP_Shape.from_transform(in_shape, ineql, biasl, inequ, biasu)


class DP_Net(nn.Sequential):
    def __init__(self, net: nn.Module):
        def to_dp_layer(layer: nn.Module, previous: Optional[nn.Module]):
            if isinstance(layer, networks.Normalization):
                return DP_Normalization(layer)
            if isinstance(layer, nn.Flatten):
                return DP_Flatten(layer)
            if isinstance(layer, nn.Linear):
                return DP_Linear(layer)
            if isinstance(layer, nn.ReLU):
                return DP_ReLU(layer)
            if isinstance(layer, networks.SPU):
                return DP_SPU(layer, previous)
            if isinstance(layer, nn.Sequential):
                # HACK: Long term solution, DP_Net should be a nn.Module with
                # a forward(...) that just "forwards" on an inner Sequential.
                return DP_Net(layer)

            raise AssertionError(f"No DP layer for layer {layer}")

        layers, previous = [], None
        for child in net.children():
            layers.append(to_dp_layer(child, previous))
            previous = child

        super().__init__(*layers)

    @staticmethod
    def _get_last_linear_layer(module: nn.Module):
        for layer in reversed(list(module.children())):
            # HACK: Again, super hack-ish. This is because we can end up with
            # nested Sequential(...) because of our hack in __init__(...) to
            # support networks.FullyConnected. But it works. ¯\_(ツ)_/¯
            if isinstance(layer, nn.Sequential):
                inner = DP_Net._get_last_linear_layer(layer)
                if inner:
                    return inner
            if isinstance(layer, DP_Linear):
                return layer
        return None

    @property
    def num_labels(self):
        linear = DP_Net._get_last_linear_layer(self)
        assert linear is not None
        return linear.out_features


class DP_Verifier(nn.Module):
    def __init__(self, num_labels: int, true_label: int):
        super().__init__()
        weight = -torch.eye(num_labels)
        weight = weight[torch.arange(num_labels) != true_label, :]
        weight[:, true_label] = torch.ones(num_labels - 1)
        self.weight = weight
        self.bias = torch.zeros(1, num_labels - 1)

    def forward(self, in_shape: DP_Shape) -> DP_Shape:
        ineql, biasl = self.weight, self.bias
        inequ, biasu = self.weight, self.bias

        out_shape = \
            DP_Shape.from_transform(in_shape, ineql, biasl, inequ, biasu)
        out_shape.backsub()
        return out_shape


class DP_Loss(nn.Module):
    def forward(self, in_shape: DP_Shape):
        lbounds = in_shape.lbounds
        return torch.log(-lbounds[lbounds < 0]).max()
