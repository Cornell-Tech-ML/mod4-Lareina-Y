from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off

max_reduce = FastOps.reduce(operators.max, -float("inf"))


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    input = input.contiguous()
    new_height = height // kh
    new_width = width // kw

    reshaped_input = input.view(batch, channel, new_height, kh, new_width, kw)
    tiled = reshaped_input.permute(0, 1, 2, 4, 3, 5).contiguous()
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width).
        kernel: Tuple (kernel_height, kernel_width).

    Returns:
    -------
        Tensor of shape (batch, channel, new_height, new_width).

    """
    tiled, new_height, new_width = tile(input, kernel)
    output = tiled.mean(4)
    return output.view(input.shape[0], input.shape[1], new_height, new_width)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input: Input tensor.
        dim: Dimension to perform argmax on.

    Returns:
    -------
        1-hot encoded tensor of the same shape as input.

    """
    max_vals = max_reduce(input, dim)
    return input == max_vals


class Max(Function):
    """New Function for max operator."""

    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Max forward function"""
        ctx.save_for_backward(input, dim)
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor, float]:
        """Max backward function"""
        input, dim = ctx.saved_values
        out = argmax(input, int(dim.item()))
        return out * d_output, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction.

    Args:
    ----
        input: Input tensor.
        dim: Dimension to reduce.

    Returns:
    -------
        Reduced tensor.

    """
    return Max.apply(input, tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor.

    Args:
    ----
        input: Input tensor.
        dim: Dimension to apply softmax on.

    Returns:
    -------
        Softmaxed tensor.

    """
    exp_vals = input.exp()
    return exp_vals / exp_vals.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor.

    Args:
    ----
        input: Input tensor.
        dim: Dimension to apply logsoftmax on.

    Returns:
    -------
        Log softmaxed tensor.

    """
    max_val = max(input, dim)
    return input - max_val - (input - max_val).exp().sum(dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width).
        kernel: Tuple (kernel_height, kernel_width).

    Returns:
    -------
        Tensor of shape (batch, channel, new_height, new_width).

    """
    tiled, new_height, new_width = tile(input, kernel)
    return max(tiled, 4).view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input: Input tensor.
        p: Dropout probability.
        ignore: Whether ignored

    Returns:
    -------
        Tensor with dropped out values.

    """
    if ignore:
        return input

    return input * (rand(input.shape) > p)
