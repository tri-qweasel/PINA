"""Module for the polynomial block model class."""

from itertools import product
import torch
from pina._src.core.utils import check_positive_integer


def generate_powers_unmixed(input_dimension, degree):
    """
    Generate exponent tuples without mixed terms.
    Example (input_dimension=2, degree=2):
        (0,0), (1,0), (0,1), (2,0), (0,2)
    """
    powers = [(0,) * input_dimension]

    for d in range(1, degree + 1):
        for i in range(input_dimension):
            exponent = [0] * input_dimension
            exponent[i] = d
            powers.append(tuple(exponent))

    return powers


def generate_powers_mixed(input_dimension, degree):
    """
    Generate all exponent tuples whose total degree is <= degree.
    Example (input_dimension=2, degree=2):
        (0,0), (1,0), (0,1), (2,0), (1,1), (0,2)
    """
    return [
        exponents
        for exponents in product(
            range(degree + 1),
            repeat=input_dimension,
        )
        if sum(exponents) <= degree
    ]


class PolynomialBlock(torch.nn.Module):
    r"""
    This layer constructs polynomial features consisting of the constant term
    and powers of each input variable up to a specified degree, then learns
    a linear combination of these features.
    The layer computes:
        y = Theta(x) * W
    where:
    - x is the input.
    - Θ(x) is the polynomial feature matrix.
    - W is a trainable coefficient matrix.
    """

    def __init__(
        self,
        degree,
        input_dimension,
        output_dimension,
        mix_dimensions=False,
    ):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree.
        input_dimension : int
            Number of input variables.
        output_dimension : int
            Number of output variables.
        mix_dimensions : bool, default=False
            If True, include mixed polynomial terms
            (e.g. x₁x₂, x₁²x₂).
            Otherwise only include powers of individual variables.
        """
        super().__init__()

        check_positive_integer(degree, strict=True)
        check_positive_integer(input_dimension, strict=True)
        check_positive_integer(output_dimension, strict=True)

        self._degree = degree
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._mix_dimensions = mix_dimensions

        if mix_dimensions:
            self._powers = generate_powers_mixed(
                input_dimension,
                degree,
            )
        else:
            self._powers = generate_powers_unmixed(
                input_dimension,
                degree,
            )

        self._linear = torch.nn.Linear(
            len(self._powers),
            output_dimension,
            bias=False,
        )

    def forward(self, x):
        """
        Parameters
        x : Tensor of shape (..., input_dimension).

        Returns
        Tensor of shape (..., output_dimension).
        """

        #check valid input_dimension
        if x.shape[-1] != self._input_dimension:
            raise ValueError(
                f"Expected input dimension {self._input_dimension}, "
                f"got {x.shape[-1]}."
            )


        theta = []

        for exponents in self._powers:
            term = torch.ones_like(x[..., 0])

            for i, p in enumerate(exponents):
                term *= x[..., i] ** p

            theta.append(term)

        theta = torch.stack(theta, dim=-1)

        return self._linear(theta)

    @property
    def degree(self):
        """Maximum polynomial degree."""
        return self._degree

    @property
    def input_dimension(self):
        """Input dimension."""
        return self._input_dimension

    @property
    def output_dimension(self):
        """Output dimension."""
        return self._output_dimension

    @property
    def mix_dimensions(self):
        """Whether mixed polynomial terms are included."""
        return self._mix_dimensions

    @property
    def coefficients(self):
        """Coefficient matrix."""
        return self._linear.weight.T

    @property
    def n_features(self):
        """Number of polynomials"""
        return len(self._powers)