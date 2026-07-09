"""Module for the polynomial block model class."""

from itertools import product
import torch
from pina._src.core.utils import check_positive_integer, check_consistency

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
        check_consistency(mix_dimensions, bool)

        self.degree = degree
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.mix_dimensions = mix_dimensions

        if self.mix_dimensions:
            powers = self.generate_powers_mixed()
        else:
            powers = self.generate_powers_unmixed()

        self.register_buffer(
            "_powers", 
            torch.tensor(powers, dtype=torch.long),
            )

        self._linear = torch.nn.Linear(
            len(self._powers),
            self.output_dimension,
            bias=False,
        )
    
    def forward(self, x):
        """
        Parameters
        x : Tensor of shape (..., input_dimension).

        Returns
        Tensor of shape (..., output_dimension).
        """
        theta = torch.prod( 
            x.unsqueeze(-2) ** self._powers, 
            dim=-1,
        )
        return self._linear(theta)
    
    def generate_powers_unmixed(self):
        """
        Generate exponent tuples without mixed terms.
        Example (input_dimension=2, degree=2):
        (0,0), (1,0), (0,1), (2,0), (0,2)
        """
        powers = [(0,) * self.input_dimension]

        for d in range(1, self.degree + 1):
            for i in range(self.input_dimension):
                exponent = [0] * self.input_dimension
                exponent[i] = d
                powers.append(tuple(exponent))

        return powers


    def generate_powers_mixed(self):
        """
        Generate all exponent tuples whose total degree is <= degree.
        Example (input_dimension=2, degree=2):
        (0,0), (1,0), (0,1), (2,0), (1,1), (0,2)
        """
        return [
            exponents
            for exponents in product(
                range(self.degree + 1),
                repeat=self.input_dimension,
            )
            if sum(exponents) <= self.degree
        ]

    @property
    def coefficients(self):
        """Coefficient matrix."""
        return self._linear.weight.T

    @property
    def n_features(self):
        """Number of polynomials"""
        return len(self._powers)