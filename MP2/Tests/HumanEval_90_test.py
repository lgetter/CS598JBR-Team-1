import pytest
from HumanEval_90 import next_smallest  # replace with your actual module name


def test_next_smallest_typical():
    assert next_smallest([2, 1, 4, 3]) == 2
    assert next_smallest([5, 3, 7, 4, 8, 2]) == 3


def test_next_smallest_edge_cases():
    assert next_smallest([]) is None
    assert next_smallest([1]) is None
    assert next_smallest([2, 2]) is None


def test_next_smallest_boundary_conditions():
    assert next_smallest([1, 2, 3]) == 2
    assert next_smallest([3, 2, 1]) == 2
    assert next_smallest([1, 1, 2, 2]) is None


def test_next_smallest_different_data_types():
    assert next_smallest([1, 2.5, 3]) == 2.5
    assert next_smallest(['a', 'b', 'c']) is None


def test_next_smallest_conditional_branches():
    assert next_smallest([1, 2]) == 2
    assert next_smallest([2, 1]) == 2
    assert next_smallest([1]) is None


def test_next_smallest_loop_iterations():
    assert next_smallest([]) is None
    assert next_smallest([1]) is None
    assert next_smallest([1, 2, 3]) == 2


def test_next_smallest_error_cases():
    with pytest.raises(TypeError):
        next_smallest(123)
    with pytest.raises(TypeError):
        next_smallest('abc')