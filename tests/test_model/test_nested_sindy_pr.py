import torch
import pytest
from math import comb
from pina.model import NestedSINDyPR
from pina.model.block import PolynomialBlock
from pina.model import SINDy

# Define a simple library of candidate functions
library = [lambda x: torch.pow(x, 2), lambda x: torch.sin(x)]


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("input_dimension", [3, 4])
@pytest.mark.parametrize("output_dimension", [5, 8])
@pytest.mark.parametrize("mix_dimensions", [False, True])
@pytest.mark.parametrize("hidden_dimension", [3,5])

def test_constructor(
    degree,
    input_dimension,
    output_dimension,
    hidden_dimension,
    mix_dimensions,
):
    model = NestedSINDyPR(
        input_dimension=input_dimension,
        degree=degree,
        library=library,
        output_dimension=output_dimension,
        hidden_dimension = hidden_dimension,
        mix_dimensions=mix_dimensions,
    )

    #check features
    if mix_dimensions:
        expected = comb(input_dimension + degree, degree)
    else:
        expected = 1 + degree * input_dimension

    assert model.polynomial_block.n_features == expected
    assert model.sindy.coefficients.shape == (len(library), output_dimension)

    # Should fail if degree is negative
    with pytest.raises(AssertionError):
        NestedSINDyPR(
            input_dimension=input_dimension,
            degree=-1,
            library=library,
            output_dimension=output_dimension,
            mix_dimensions=mix_dimensions,
            hidden_dimension=hidden_dimension,
        )

    # Should fail if input_dimension is negative
    with pytest.raises(AssertionError):
        NestedSINDyPR(
            input_dimension=-1,
            degree=degree,
            library=library,
            output_dimension=output_dimension,
            mix_dimensions=mix_dimensions,
            hidden_dimension=hidden_dimension,
        )

    # Should fail if output_dimension is negative
    with pytest.raises(AssertionError):
        NestedSINDyPR(
            input_dimension=input_dimension,
            degree=degree,
            library=library,
            output_dimension=-1,
            mix_dimensions=mix_dimensions,
            hidden_dimension=hidden_dimension,
        )

    # Should fail if mix_dimensions is not a bool
    with pytest.raises(ValueError):
        NestedSINDyPR(
            input_dimension=input_dimension,
            degree=degree,
            library=library,
            output_dimension=output_dimension,
            mix_dimensions="True",
            hidden_dimension=hidden_dimension,
        )

    # Should fail if library is not a list
    with pytest.raises(ValueError):
        NestedSINDyPR(
            input_dimension=input_dimension,
            degree=degree,
            library=lambda x: torch.pow(x, 2),
            output_dimension=output_dimension,
            mix_dimensions=mix_dimensions,
            hidden_dimension=hidden_dimension,
        )

    # Should fail if library is not a list of callables
    with pytest.raises(ValueError):
        NestedSINDyPR(
            input_dimension=input_dimension,
            degree=degree,
            library=[1, 2, 3],
            output_dimension=output_dimension,
            mix_dimensions=mix_dimensions,
            hidden_dimension=hidden_dimension,
        )
    
    # Should fail if hidden_dimension is negative 
    with pytest.raises(AssertionError):
        NestedSINDyPR(
            input_dimension=input_dimension,
            degree=degree,
            library=library,
            output_dimension=output_dimension,
            mix_dimensions=mix_dimensions,
            hidden_dimension=-1,
        )


@pytest.mark.parametrize("data", [torch.rand((20, 3)), torch.rand((5, 20, 4))])
def test_forward(data):

    model = NestedSINDyPR(
        input_dimension=data.shape[-1],
        degree=2,
        library=library,
        output_dimension=data.shape[-1],
        hidden_dimension = 3,
    )
    output_ = model(data)
    
    assert output_.shape == data.shape


@pytest.mark.parametrize("data", [torch.rand((20, 3)), torch.rand((5, 20, 4))])
def test_backward(data):

    model = NestedSINDyPR(
        input_dimension=data.shape[-1],
        degree=2,
        library=library,
        output_dimension=data.shape[-1],
        hidden_dimension = 3,
    )
    output_ = model(data.requires_grad_())

    loss = output_.mean()
    loss.backward()
    assert data.grad.shape == data.shape