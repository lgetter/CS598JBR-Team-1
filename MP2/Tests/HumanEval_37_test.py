import pytest
from HumanEval_37 import sort_even  # replace with your actual module name


def test_sort_even_normal():
    assert sort_even([1, 2, 3, 4, 5, 6, 7, 8]) == [2, 1, 4, 3, 6, 5, 8, 7]


def test_sort_even_empty():
    assert sort_even([]) == []


def test_sort_even_single_element():
    assert sort_even([1]) == [1]


def test_sort_even_max_values():
    assert sort_even([100, 200, 300, 400, 500, 600, 700, 800]) == [200, 100, 400, 300, 600, 500, 800, 700]


def test_sort_even_boundary_conditions():
    assert sort_even([0, 1, 2, 3, 4, 5, 6, 7]) == [2, 0, 4, 1, 6, 3, 7, 5]


def test_sort_even_different_data_types():
    assert sort_even([1, 'b', 3.5, 'd', 5, 'f', 7.2, 'h']) == ['b', 1, 'd', 3.5, 'f', 5, 'h', 7.2]


def test_sort_even_conditional_branches():
    assert sort_even([1, 2, 3, 4, 5, 6, 7]) == [2, 1, 4, 3, 6, 5, 7]


def test_sort_even_loop_iterations():
    assert sort_even([1]) == [1]
    assert sort_even([]) == []
    assert sort_even([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [2, 1, 4, 3, 6, 5, 8, 7, 10, 9]


def test_sort_even_error_cases():
    with pytest.raises(TypeError):
        sort_even('invalid input')
    with pytest.raises(TypeError):
        sort_even([1, 'b', 'c'])