import pytest
from HumanEval_152 import compare  # replace with your actual module name

def test_compare_normal():
    assert compare([1, 2, 3], [1, 2, 3]) == [0, 0, 0]
    assert compare([4, 5, 6], [1, 2, 3]) == [3, 3, 3]
    assert compare([1, 1, 1], [0, 0, 0]) == [1, 1, 1]

def test_compare_edge_cases():
    assert compare([], []) == []
    assert compare([1], [1]) == [0]
    assert compare([100], [100]) == [0]

def test_compare_boundary_conditions():
    assert compare([10, 10, 10], [0, 0, 0]) == [10, 10, 10]
    assert compare([0, 0, 0], [10, 10, 10]) == [10, 10, 10]

def test_compare_different_data_types():
    assert compare([1, 2, 3], [1.0, 2.0, 3.0]) == [0, 0, 0]
    assert compare([1, 2, 3], ['1', '2', '3']) == [0, 0, 0]

def test_compare_conditional_branches():
    assert compare([1, 2, 3], [1, 2, 3]) == [0, 0, 0]
    assert compare([1, 2, 3], [4, 5, 6]) == [3, 3, 3]

def test_compare_loop_iterations():
    assert compare([], []) == []
    assert compare([1], [1]) == [0]
    assert compare([1, 2, 3], [1, 2, 3]) == [0, 0, 0]

def test_compare_error_cases():
    with pytest.raises(TypeError):
        compare(123, 456)
    with pytest.raises(ValueError):
        compare(['a', 'b', 'c'], ['d', 'e', 'f'])