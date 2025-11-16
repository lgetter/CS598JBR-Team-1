import pytest
from HumanEval_142 import sum_squares  # replace with your actual module name


def test_sum_squares_typical_inputs():
    assert sum_squares([1, 2, 3, 4, 5]) == 31
    assert sum_squares([10, 20, 30, 40, 50]) == 1500


def test_sum_squares_edge_cases():
    assert sum_squares([]) == 0
    assert sum_squares([1]) == 1
    assert sum_squares([100]) == 100


def test_sum_squares_boundary_conditions():
    assert sum_squares([1000]*1001) == 1001000


def test_sum_squares_different_data_types():
    assert sum_squares([1.5, 2.5, 3.5, 4.5, 5.5]) == 41.25
    assert sum_squares([-1, -2, -3, -4, -5]) == -15


def test_sum_squares_conditional_branches():
    assert sum_squares([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 29


def test_sum_squares_loop_iterations():
    assert sum_squares([]) == 0
    assert sum_squares([1]) == 1
    assert sum_squares([1, 2, 3]) == 10


def test_sum_squares_error_cases():
    with pytest.raises(TypeError):
        sum_squares('1, 2, 3')
    with pytest.raises(TypeError):
        sum_squares(None)