import torch
import pytest
from math import comb
from pina.model import NestedSINDyPR

# Simple library of candidate functions
library = [lambda x: torch.pow(x, 2), lambda x: torch.sin(x)]


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("input_dimension", [2, 3])
@pytest.mark.parametrize("output_dimension", [2, 3])
@pytest.mark.parametrize("mix_dimensions", [False, True])
def test_constructor(
    degree,
    input_dimension,
    output_dimension,
    mix_dimensions,
):
    model = NestedSINDyPR(
        input_dimension=input_dimension,
        degree=degree,
        library=library,
        output_dimension=output_dimension,
        mix_dimensions=mix_dimensions,
    )

    # Check polynomial features
    if mix_dimensions:
        expected = comb(input_dimension + degree, degree)
    else:
        expected = 1 + degree * input_dimension

    assert model.polynomial_block.n_features == expected
    assert model.library_length == len(library)
    assert model.hidden_dimension == output_dimension * len(library)
    assert model.coefficients.shape == (output_dimension, len(library))

    # Should fail if output_dimension is not a positive integer
    with pytest.raises(AssertionError):
        NestedSINDyPR(
            input_dimension=input_dimension,
            degree=degree,
            library=library,
            output_dimension=-1,
            mix_dimensions=mix_dimensions,
        )

    # Should fail if library is not a list
    with pytest.raises(ValueError):
        NestedSINDyPR(
            input_dimension=input_dimension,
            degree=degree,
            library=lambda x: torch.pow(x, 2),
            output_dimension=output_dimension,
            mix_dimensions=mix_dimensions,
        )

    # Should fail if library is not a list of callables
    with pytest.raises(ValueError):
        NestedSINDyPR(
            input_dimension=input_dimension,
            degree=degree,
            library=[1, 2, 3],
            output_dimension=output_dimension,
            mix_dimensions=mix_dimensions,
        )


@pytest.mark.parametrize("data", [torch.rand((20, 2)), torch.rand((5, 20, 2))])
def test_forward(data):
    model = NestedSINDyPR(
        input_dimension=data.shape[-1],
        degree=2,
        library=library,
        output_dimension=data.shape[-1],
        mix_dimensions=False,
    )

    output_ = model(data)

    assert output_.shape == data.shape
    assert torch.isfinite(output_).all()


@pytest.mark.parametrize("data", [torch.rand((20, 2)), torch.rand((5, 20, 2))])
def test_backward(data):
    model = NestedSINDyPR(
        input_dimension=data.shape[-1],
        degree=2,
        library=library,
        output_dimension=data.shape[-1],
        mix_dimensions=False,
    )

    output_ = model(data.requires_grad_())
    loss = output_.mean()
    loss.backward()

    assert data.grad is not None
    assert data.grad.shape == data.shape