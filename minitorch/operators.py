"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """Multiplies two numbers

    Args:
    ----
        x: First float number
        y: Second float number

    Returns:
    -------
        The product of x and y

    """
    return x * y


def id(x: float) -> float:
    """Get the input unchanged

    Args:
    ----
        x: A float number

    Returns:
    -------
        The same input number

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers

    Args:
    ----
        x: First float number
        y: Second float number

    Returns:
    -------
        The sum of two input numbers

    """
    return x + y


def neg(x: float) -> float:
    """Negates a number

    Args:
    ----
        x: A float number

    Returns:
    -------
        The negation of the input number

    """
    return -x


def lt(x: float, y: float) -> float:
    """Checks if one number is less than another

    Args:
    ----
        x: First float number
        y: Second float number

    Returns:
    -------
        True if x is less than y, otherwise False

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal

    Args:
    ----
        x: First float number
        y: Second float number

    Returns:
    -------
        True if x equals y, otherwise False

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Get the larger of two numbers

    Args:
    ----
        x: First float number
        y: Second float number

    Returns:
    -------
        The greater number between x and y

    """
    return x if x > y else y


# For is_close:
# $f(x) = |x - y| < 1e-2$
def is_close(x: float, y: float) -> float:
    """Checks if two numbers are close in value

    Args:
    ----
        x: First float number
        y: Second float number

    Returns:
    -------
        True if the difference between two numbers is less than 1e-2, otherwise False

    """
    return 1.0 if abs(x - y) < 1e-2 else 0.0


# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
def sigmoid(x: float) -> float:
    """Calculates the sigmoid function

    Args:
    ----
        x: A float number

    Returns:
    -------
        The sigmoid of the input

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function

    Args:
    ----
        x: A float number

    Returns:
    -------
        The ReLU of the input number. 0 when x is negative, otherwise x

    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculates the natural logarithm

    Args:
    ----
        x: A float number

    Returns:
    -------
        The natural logarithm of input number

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculates the exponential function

    Args:
    ----
        x: A float number

    Returns:
    -------
        The exponential of input number

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal

    Args:
    ----
        x: A float number

    Returns:
    -------
        The reciprocal of input number

    """
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Computes the derivative of log times a second arg

    Args:
    ----
        x: A float number
        d: Derivative number

    Returns:
    -------
        The backpropagated derivative of log

    """
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second arg

    Args:
    ----
        x: A float number
        d: Derivative number

    Returns:
    -------
        The backpropagation of the inverse function

    """
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of ReLU times a second arg

    Args:
    ----
        x: A float number
        d: Derivative number

    Returns:
    -------
        The backpropagation of the ReLU function.

    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(func: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a given function to each element of an iterable

    Args:
    ----
        func: Function from one value to one value

    Returns:
    -------
        A function that takes a list, applies `fn` to each element, and returns a new list

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(func(x))
        return ret

    return _map


def zipWith(
    func: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Combines elements from two iterables using a given function

    Args:
    ----
        func: combine two values

    Returns:
    -------
        A function that takes two equally sized lists `ls1` and `ls2`,
        produce a new list by applying fn(x, y) on each pair of elements.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(func(x, y))
        return ret

    return _zipWith


def reduce(
    func: Callable[[float, float], float], init: float = 0.0
) -> Callable[[Iterable[float]], float]:
    """Reduces an iterable to a single value using a given function

    Args:
    ----
        func: Function to apply
        iterable: Iterable of numbers
        init: Initial value, default to be 0.0

    Returns:
    -------
        Function that takes a list `ls` of elements
        $x_1 dots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`

    """

    def _reduce(iterable: Iterable[float]) -> float:
        result = init
        for i in iterable:
            result = func(result, i)
        return result

    return _reduce


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map

    Args:
    ----
        ls: List of numbers

    Returns:
    -------
        A list where each element is negated

    """
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith

    Args:
    ----
        ls1: First list of numbers
        ls2: Second list of numbers

    Returns:
    -------
        A list where each element is the sum of the corresponding elements in ls1 and ls2

    """
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list using reduce

    Args:
    ----
        ls: List of numbers

    Returns:
    -------
        The sum of the elements in the list

    """
    return reduce(add)(ls)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce

    Args:
    ----
        ls: List of numbers

    Returns:
    -------
        The product of the elements in the list

    """
    return reduce(mul, 1.0)(ls)
