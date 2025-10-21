import pytest
from HumanEval_62 import derivative

def test_derivative_typical_inputs():
    assert derivative([1, 2, 3, 4]) == [2, 6, 12]
    assert derivative([0, 1, 2, 3]) == [1, 4]

def test_derivative_edge_cases():
    assert derivative([]) == []
    assert derivative([1]) == []
    assert derivative([1, 2]) == [2]

def test_derivative_boundary_conditions():
    assert derivative([10]) == []
    assert derivative([-1, -2, -3, -4]) == [-2, -6, -12]

def test_derivative_different_data_types():
    assert derivative([1.0, 2.0, 3.0]) == [2.0, 6.0]
    assert derivative([1, 2, 3, 4.0]) == [2.0, 6.0, 12.0]

def test_derivative_conditional_branches():
    assert derivative([0, 1, 2, 3]) == [1, 4]
    assert derivative([10]) == []

def test_derivative_loop_iterations():
    assert derivative([]) == []
    assert derivative([1]) == []
    assert derivative([1, 2, 3, 4]) == [2, 6, 12]

def test_derivative_error_cases():
    with pytest.raises(TypeError):
        derivative(1)
    with pytest.raises(TypeError):
        derivative('1')
    with pytest.raises(TypeError):
        derivative(None)