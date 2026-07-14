"""Module for the NestedSINDyPR model class."""

import torch
from pina.model.block import PolynomialBlock
from pina.model import SINDy
from pina._src.core.utils import check_positive_integer


class NestedSINDyPR(torch.nn.Module):
    r"""
    NestedSINDyPR model class.

    This model combines:
      1. A PolynomialBlock, which maps the input state to a latent
         polynomial representation.
      2. A SINDy block, which performs sparse regression on the output
         of the polynomial block.
    """

    def __init__(
        self,
        input_dimension,
        degree,
        library,
        output_dimension,
        hidden_dimension,
        mix_dimensions=False,
    ):
        """
        Initialization of the :class:`NestedSINDyPR` class.

        :param int input_dimension: The number of input variables of the
            original state.
        :param int degree: The maximum polynomial degree used in the
            :class:`PolynomialBlock`.
        :param list[Callable] library: The collection of candidate functions
            used by the :class:`SINDy` block. Each function must accept an
            input tensor of shape ``[..., hidden_dimension]`` and return a
            tensor of shape ``[..., 1]``.
        :param int output_dimension: The number of output variables of the
            final model.
        :param int hidden_dimension: The number of latent variables produced
            by the :class:`PolynomialBlock`. 
        :param bool mix_dimensions: If ``True``, include mixed polynomial
            terms in the :class:`PolynomialBlock`. Otherwise only powers of
            individual variables are used.
        """
        super().__init__()
        check_positive_integer(hidden_dimension, strict=True)

        self.polynomial_block = PolynomialBlock(
            degree=degree,
            input_dimension=input_dimension,
            output_dimension=hidden_dimension,
            mix_dimensions=mix_dimensions,
        )

        self.sindy = SINDy(
            library=library,
            output_dimension=output_dimension,
        )

    

    def forward(self, x):
        """
        Forward pass of the :class:`NestedSINDyPR` model.

        :param torch.Tensor x: The input batch of state variables.
        :return: The predicted output of the nested model.
        :rtype: torch.Tensor
        """
        polynomial = self.polynomial_block(x)
        return self.sindy(polynomial)
