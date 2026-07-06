import torch
import pytest
from pina.model.block import PolynomialBlock

data = torch.rand((20, 3))


@pytest.mark.parametrize(
    "degree,input_dimension,output_dimension,mix_dimensions",
    [
        (2, 3, 5, False),
        (2, 3, 5, True),
        (3, 3, 8, False),
        (3, 3, 8, True),
    ],
)
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


@pytest.mark.parametrize(
    "degree,output_dimension,mix_dimensions",
    [
        (2, 5, False),
        (2, 5, True),
        (3, 8, False),
        (3, 8, True),
    ],
)
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


@pytest.mark.parametrize(
    "degree,output_dimension,mix_dimensions",
    [
        (2, 5, False),
        (2, 5, True),
        (3, 8, False),
        (3, 8, True),
    ],
)
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

    assert x.grad is not None
    assert x.grad.shape == x.shape


@pytest.mark.parametrize(
    "mix_dimensions",
    [
        False,
        True,
    ],
)
def test_input_dimension_check(mix_dimensions):

    model = PolynomialBlock(
        degree=2,
        input_dimension=3,
        output_dimension=5,
        mix_dimensions=mix_dimensions,
    )

    wrong_data = torch.rand((20, 2))

    with pytest.raises(ValueError):
        model(wrong_data)