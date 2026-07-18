"""Module for the NestedSINDy model class."""

from typing import Callable
import torch
from pina.model.block import PolynomialBlock
from pina._src.core.utils import check_positive_integer, check_consistency


class NestedSINDy(torch.nn.Module):
    r"""
    NestedSINDy model class.

    Architecture:
    - use_prp=False:
        x -> first PR stage -> y

      In this case, `pr_output_dim` is the final output dimension of the model.

    - use_prp=True:
        x -> first PR stage -> y -> PolynomialBlock -> output

      In this case:
        - `pr_output_dim` is the output dimension of the first PR stage
        - `prp_output_dim` is the final output dimension of the model

    Notes:
    - The `library` is used only in the first PR stage.
    - The second `PolynomialBlock` directly returns the final output tensor.
    - If `prp_output_dim` is not provided and `use_prp=True`,
      it defaults to `pr_output_dim`.
    """

    def __init__(
        self,
        input_dim,
        degree,
        library,
        pr_output_dim,
        mix_dimensions=False,
        use_prp=False,
        prp_degree=None,
        prp_output_dim=None,
    ):
        """
        Initialization of the :class:`NestedSINDy` class.

        :param int input_dimension: Number of input variables.
        :param int degree: Maximum polynomial degree used in the first PolynomialBlock.
        :param list[Callable] library: Candidate functions applied to the latent polynomial features in the first PR stage.
        :param int pr_output_dim: Final output dimension if use_prp=False, or output dimension of the first PR stage if use_prp=True.
        :param bool mix_dimensions: If True, include mixed polynomial terms.
        :param bool use_prp: If True, enable the second polynomial stage.
        :param int prp_degree: Maximum polynomial degree used in the second PolynomialBlock.
        :param int prp_output_dim: Final output dimension if use_prp=True. Defaults to pr_output_dim.
        """
        super().__init__()

        if not isinstance(library, list):
            raise ValueError("`library` must be a list of callables.")
        check_consistency(library, Callable)
        check_positive_integer(pr_output_dim, strict=True)

        self.library = library
        self.library_length = len(self.library)

        self.use_prp = use_prp
        self.pr_output_dim = pr_output_dim
        self.prp_degree = degree if prp_degree is None else prp_degree

        # If PRP is enabled, the second stage has its own final output dimension.
        # If not provided, default to pr_output_dim.
        if self.use_prp:
            self.prp_output_dim = (
                self.pr_output_dim if prp_output_dim is None else prp_output_dim
            )
            check_positive_integer(self.prp_output_dim, strict=True)
        else:
            self.prp_output_dim = None

        # First stage hidden size:
        # one block of len(library) features per PR output dimension
        self.hidden_dimension1 = self.pr_output_dim * self.library_length

        self.polynomial_block1 = PolynomialBlock(
            degree=degree,
            input_dimension= input_dim,
            output_dimension=self.hidden_dimension1,
            mix_dimensions=mix_dimensions,
        )

        # PR coefficients
        self.coefficients1 = torch.nn.Parameter(
            0.01 * torch.randn(self.pr_output_dim, self.library_length)
        )

        # Optional PRP stage:
        # The second PolynomialBlock directly returns the final output tensor,
        # so no reshape/library function is needed here.
        if self.use_prp:
            self.polynomial_block2 = PolynomialBlock(
                degree=self.prp_degree,
                input_dimension=self.pr_output_dim,
                output_dimension=self.prp_output_dim,
                mix_dimensions=mix_dimensions,
            )

    def _apply_library(self, polynomial, stage_output_dim):
        """
        Apply the library functions blockwise to the polynomial features.

        :param torch.Tensor polynomial: Tensor of shape
            (..., stage_output_dim * library_length).
        :param int stage_output_dim: Output dimension of the current PR stage.
        :return: Tensor of shape (..., stage_output_dim, library_length).
        """
        polynomial = polynomial.reshape(
            *polynomial.shape[:-1],
            stage_output_dim,
            self.library_length,
        )

        theta = torch.cat(
            [func(polynomial[..., i:i + 1]) for i, func in enumerate(self.library)],
            dim=-1,
        )
        return theta

    def _pr_stage(self, x):
        """
        PR Layer
        :param torch.Tensor x: Input batch.
        :return: Tensor of shape (..., pr_output_dim)
        """
        polynomial = self.polynomial_block1(x)
        theta = self._apply_library(polynomial, self.pr_output_dim)
        output = (theta * self.coefficients1.unsqueeze(0)).sum(dim=-1)
        return output

    def forward(self, x):
        """
        Forward pass of the :class:`NestedSINDy` model.

        :param torch.Tensor x: Input batch.
        :return: Model prediction.
        :rtype: torch.Tensor
        """
        # First PR stage
        y = self._pr_stage(x)

        # If use_prp=False, this is the final output
        if not self.use_prp:
            return y

        # use_prp=True: second PolynomialBlock directly produces final output
        return self.polynomial_block2(y)