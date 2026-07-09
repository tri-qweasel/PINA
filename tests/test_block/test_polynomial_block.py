import torch
import pytest
from math import comb
from pina.model.block import PolynomialBlock

data = torch.rand((20, 3))


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("input_dimension", [3])
@pytest.mark.parametrize("output_dimension", [5, 8])
@pytest.mark.parametrize("mix_dimensions", [False, True])

def test_constructor(
    degree,
    input_dimension,
    output_dimension,
    mix_dimensions,
):

    PolynomialBlock(
        degree=degree,
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        mix_dimensions=mix_dimensions,
    )

    # Should fail if degree is negative
    with pytest.raises(AssertionError):
        PolynomialBlock(
            degree=-1,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            mix_dimensions=mix_dimensions,
        )

    # Should fail if input_dimension is negative
    with pytest.raises(AssertionError):
        PolynomialBlock(
            degree=degree,
            input_dimension=-1,
            output_dimension=output_dimension,
            mix_dimensions=mix_dimensions,
        )

    # Should fail if output_dimension is negative
    with pytest.raises(AssertionError):
        PolynomialBlock(
            degree=degree,
            input_dimension=input_dimension,
            output_dimension=-1,
            mix_dimensions=mix_dimensions,
        )
    
    # Should fail if mix_dimensions is not a bool
    with pytest.raises(ValueError):
        PolynomialBlock(
            degree=degree,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            mix_dimensions="True",
        )
    


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("output_dimension", [5, 8])
@pytest.mark.parametrize("mix_dimensions", [False, True])

def test_forward(
    degree,
    output_dimension,
    mix_dimensions,
):

    model = PolynomialBlock(
        degree=degree,
        input_dimension=data.shape[-1],
        output_dimension=output_dimension,
        mix_dimensions=mix_dimensions,
    )

    output_ = model(data)

    assert output_.shape == (data.shape[0], output_dimension)


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("output_dimension", [5, 8])
@pytest.mark.parametrize("mix_dimensions", [False, True])

def test_backward(
    degree,
    output_dimension,
    mix_dimensions,
):

    model = PolynomialBlock(
        degree=degree,
        input_dimension=data.shape[-1],
        output_dimension=output_dimension,
        mix_dimensions=mix_dimensions,
    )

    x = data.clone().detach().requires_grad_(True)

    output_ = model(x)

    loss = torch.mean(output_)
    loss.backward()

    assert x.grad.shape == x.shape

@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("input_dimension", [3])
@pytest.mark.parametrize("mix_dimensions", [False, True])

def test_n_features(
    degree,
    input_dimension,
    mix_dimensions,
):
    model = PolynomialBlock(
        degree=degree,
        input_dimension=input_dimension,
        output_dimension=5,
        mix_dimensions=mix_dimensions,
    )

    if mix_dimensions:
        expected = comb(input_dimension + degree, degree)
    else:
        expected = 1 + degree * input_dimension

    assert model.n_features == expected