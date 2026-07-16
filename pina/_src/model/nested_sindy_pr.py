"""Module for the NestedSINDyPR model class."""

from typing import Callable
import torch
from pina.model.block import PolynomialBlock
from pina._src.core.utils import check_positive_integer, check_consistency


class NestedSINDyPR(torch.nn.Module):
    r"""
    NestedSINDyPR model class.

    This model combines:
      1. A PolynomialBlock, which maps the input state to a latent
         polynomial representation.
      2. A learned library expansion, applied blockwise per output.
      3. A trainable linear readout.
    """

    def __init__(
        self,
        input_dimension,
        degree,
        library,
        output_dimension,
        mix_dimensions=False,
    ):
        """
        Initialization of the :class:`NestedSINDyPR` class.

        :param int input_dimension: Number of input variables.
        :param int degree: Maximum polynomial degree used in PolynomialBlock.
        :param list[Callable] library: Candidate functions applied to the
            latent polynomial features.
        :param int output_dimension: Number of output variables.
        :param bool mix_dimensions: If True, include mixed polynomial terms.
        """
        super().__init__()

        check_positive_integer(output_dimension, strict=True)
        check_consistency(library, Callable)
        if not isinstance(library, list):
            raise ValueError("`library` must be a list of callables.")

        self.library = library
        self.library_length = len(self.library)
        self.output_dimension = output_dimension

        # Exactly enough latent features:
        # one block of len(library) features per output
        self.hidden_dimension = self.output_dimension * self.library_length

        self.polynomial_block = PolynomialBlock(
            degree=degree,
            input_dimension=input_dimension,
            output_dimension=self.hidden_dimension,
            mix_dimensions=mix_dimensions,
        )

        # Coefficients are output-specific:
        # shape = (output_dimension, library_length)
        self.coefficients = torch.nn.Parameter( 0.01 * torch.randn(output_dimension, self.library_length))

    def forward(self, x):
        """
        Forward pass of the :class:`NestedSINDyPR` model.

        :param torch.Tensor x: Input batch.
        :return: Model prediction with shape [..., output_dimension].
        :rtype: torch.Tensor
        """
        # polynomial shape: (..., output_dimension * library_length)
        polynomial = self.polynomial_block(x)

        # reshape into one block per output:
        # (..., output_dimension, library_length)
        polynomial = polynomial.reshape(
            *polynomial.shape[:-1],
            self.output_dimension,
            self.library_length,
        )

        # Apply each library function to its corresponding latent feature
        # across all outputs.
        # theta shape: (..., output_dimension, library_length)
        theta = torch.cat(
            [func(polynomial[..., i:i+1]) for i, func in enumerate(self.library)],
            dim=-1,
        )

        # Weighted sum over library dimension
        # coefficients shape: (output_dimension, library_length)
        # output shape: (..., output_dimension)
        output = (theta * self.coefficients.unsqueeze(0)).sum(dim=-1)
        return output

