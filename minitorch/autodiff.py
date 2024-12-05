from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_pos = list(vals)
    vals_neg = list(vals)

    vals_pos[arg] += epsilon
    vals_neg[arg] -= epsilon

    return (f(*vals_pos) - f(*vals_neg)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate a derivative value in the variable

        Args:
        ----
            x: The value to accumulate

        Returns:
        -------
            None

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for this variable"""
        ...

    def is_leaf(self) -> bool:
        """Check if this variable is the last one (no `last_fn`)"""
        ...

    def is_constant(self) -> bool:
        """Check if this variable is a constant"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of this variable"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to propagate the derivative"""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    sorted_vars = []

    def visit(var: Variable) -> None:
        if var.unique_id in visited or var.is_constant():
            return

        if not var.is_leaf():
            for parent in var.parents:
                if not parent.is_constant():
                    visit(parent)

        visited.add(var.unique_id)
        sorted_vars.append(var)

    visit(variable)
    return reversed(sorted_vars)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    topo_order = topological_sort(variable)

    dict = {}
    dict[variable.unique_id] = deriv

    for var in topo_order:
        d = dict[var.unique_id]

        if var.is_leaf():
            var.accumulate_derivative(d)
        else:
            for parent, d_output in var.chain_rule(d):
                if parent.unique_id not in dict:
                    dict[parent.unique_id] = d_output
                else:
                    dict[parent.unique_id] += d_output


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Fetch the saved values in the context during backpropagation"""
        return self.saved_values
