import pytest
import math

def test_do_algebra_normal():
    assert do_algebra(['+', '-', '*', '/'], [3, 4, 5, 2]) == 1

def test_do_algebra_edge_cases():
    assert do_algebra(['+', '-'], [3, 4]) == 7
    assert do_algebra(['*'], [3, 4]) == 12
    assert do_algebra(['/'], [3, 1]) == 3
    assert do_algebra([], []) == ''
    assert do_algebra(['+'], [3]) == 3

def test_do_algebra_boundary_conditions():
    assert do_algebra(['+'], [math.inf]) == math.inf
    assert do_algebra(['-'], [math.inf]) == -math.inf
    assert do_algebra(['*'], [0, 1]) == 0
    assert do_algebra(['/'], [0, 1]) == 0

def test_do_algebra_different_data_types():
    assert do_algebra(['+'], [3.0, 4]) == 7.0
    assert do_algebra(['+'], [3, 4.0]) == 7.0
    assert do_algebra(['+'], [3.0, 4.0]) == 7.0

def test_do_algebra_conditional_branches():
    assert do_algebra(['+'], [3]) == 3
    assert do_algebra(['-'], [3]) == -3
    assert do_algebra(['*'], [3]) == 0
    assert do_algebra(['/'], [3]) == 'inf'

def test_do_algebra_loop_iterations():
    assert do_algebra(['+'], []) == 0
    assert do_algebra(['+'], [3]) == 3
    assert do_algebra(['+'], [3, 4, 5]) == 12

def test_do_algebra_error_cases():
    with pytest.raises(ZeroDivisionError):
        do_algebra(['/'], [3, 0])
    with pytest.raises(TypeError):
        do_algebra(['+'], [3, '4'])
    with pytest.raises(TypeError):
        do_algebra(['+'], ['3', 4])
    with pytest.raises(TypeError):
        do_algebra(['+'], [3, None])