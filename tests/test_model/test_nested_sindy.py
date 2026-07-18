import torch
import pytest
from math import comb

from pina.model import NestedSINDy


# Simple library of candidate functions
def square(x):
    return torch.pow(x, 2)


def sine(x):
    return torch.sin(x)


library = [square, sine]


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("input_dim", [2, 3])
@pytest.mark.parametrize("pr_output_dim", [2, 3])
@pytest.mark.parametrize("mix_dimensions", [False, True])
@pytest.mark.parametrize("use_prp", [False, True])
@pytest.mark.parametrize("prp_degree", [None, 2])
def test_constructor(
    degree,
    input_dim,
    pr_output_dim,
    mix_dimensions,
    use_prp,
    prp_degree,
):
    effective_prp_degree = degree if prp_degree is None else prp_degree

    model = NestedSINDy(
        input_dim=input_dim,
        degree=degree,
        library=library,
        pr_output_dim=pr_output_dim,
        mix_dimensions=mix_dimensions,
        use_prp=use_prp,
        prp_degree=prp_degree,
    )

    # First polynomial block
    if mix_dimensions:
        expected1 = comb(input_dim + degree, degree)
    else:
        expected1 = 1 + degree * input_dim

    assert model.polynomial_block1.n_features == expected1

    # Second polynomial block only exists if use_prp=True
    if use_prp:
        if mix_dimensions:
            expected2 = comb(pr_output_dim + effective_prp_degree, effective_prp_degree)
        else:
            expected2 = 1 + effective_prp_degree * pr_output_dim

        assert model.prp_output_dim == pr_output_dim
        assert model.polynomial_block2.n_features == expected2
    else:
        assert model.prp_output_dim is None
        assert not hasattr(model, "polynomial_block2")

    # Should fail if pr_output_dim is not a positive integer
    with pytest.raises(AssertionError):
        NestedSINDy(
            input_dim=input_dim,
            degree=degree,
            library=library,
            pr_output_dim=-1,
            mix_dimensions=mix_dimensions,
        )

    # Should fail if library is not a list
    with pytest.raises(ValueError):
        NestedSINDy(
            input_dim=input_dim,
            degree=degree,
            library=lambda x: torch.pow(x, 2),
            pr_output_dim=pr_output_dim,
            mix_dimensions=mix_dimensions,
        )

    # Should fail if library is not a list of callables
    with pytest.raises(ValueError):
        NestedSINDy(
            input_dim=input_dim,
            degree=degree,
            library=[1, 2, 3],
            pr_output_dim=pr_output_dim,
            mix_dimensions=mix_dimensions,
        )


@pytest.mark.parametrize("use_prp", [False, True])
@pytest.mark.parametrize("data", [torch.rand((20, 2)), torch.rand((5, 20, 2))])
def test_forward(use_prp, data):
    model_kwargs = dict(
        input_dim=data.shape[-1],
        degree=2,
        library=library,
        pr_output_dim=data.shape[-1],
        mix_dimensions=False,
        use_prp=use_prp,
    )

    if use_prp:
        model_kwargs["prp_degree"] = 2
        model_kwargs["prp_output_dim"] = data.shape[-1]

    model = NestedSINDy(**model_kwargs)

    output_ = model(data)

    assert output_.shape == data.shape
    assert torch.isfinite(output_).all()


@pytest.mark.parametrize("use_prp", [False, True])
@pytest.mark.parametrize("data", [torch.rand((20, 2)), torch.rand((5, 20, 2))])
def test_backward(use_prp, data):
    model_kwargs = dict(
        input_dim=data.shape[-1],
        degree=2,
        library=library,
        pr_output_dim=data.shape[-1],
        mix_dimensions=False,
        use_prp=use_prp,
    )

    if use_prp:
        model_kwargs["prp_degree"] = 2
        model_kwargs["prp_output_dim"] = data.shape[-1]

    model = NestedSINDy(**model_kwargs)

    output_ = model(data.requires_grad_())
    loss = output_.mean()
    loss.backward()

    assert data.grad is not None
    assert data.grad.shape == data.shape