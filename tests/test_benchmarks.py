import torch
from rsde_opt.benchmarks import rastrigin_function, ackley_function


def test_rastrigin_function_output_shape():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    output = rastrigin_function(x)
    assert output.shape == (2,)


def test_rastrigin_function_zero_input():
    x = torch.zeros(1, 2)
    output = rastrigin_function(x)
    assert torch.allclose(output, torch.tensor([0.]))


def test_rastrigin_function_known_input():
    x = torch.tensor([[1.0, 1.0]])
    expected_output = 10 * 2 + 2 - 10 * (torch.cos(2 * torch.pi * x)).sum()
    output = rastrigin_function(x)
    assert torch.allclose(output, expected_output)


def test_ackley_function_output_shape():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    output = ackley_function(x)
    assert output.shape == (3,)


def test_ackley_function_global_minimum():
    x = torch.tensor([[2.0, 2.0]])
    output = ackley_function(x)
    assert torch.allclose(output, torch.tensor([0.]), atol=1e-6)
