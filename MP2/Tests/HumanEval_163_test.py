import pytest
from HumanEval_163 import generate_integers  # replace with your actual module name


def test_generate_integers_normal():
    assert generate_integers(5, 7) == [6]
    assert generate_integers(3, 8) == [4, 6]
    assert generate_integers(2, 8) == [4, 6]
    assert generate_integers(3, 9) == [4, 6, 8]


def test_generate_integers_edge_cases():
    assert generate_integers(1, 1) == []
    assert generate_integers(1, 2) == []
    assert generate_integers(8, 8) == []
    assert generate_integers(9, 9) == []


def test_generate_integers_boundary_conditions():
    assert generate_integers(1, 8) == [4, 6]
    assert generate_integers(2, 9) == [4, 6, 8]
    assert generate_integers(1, 10) == [4, 6, 8]
    assert generate_integers(9, 10) == [10]


def test_generate_integers_different_data_types():
    assert generate_integers(5.5, 7.7) == [6]
    assert generate_integers(3.3, 8.8) == [4, 6]
    assert generate_integers(2.2, 8.8) == [4, 6]
    assert generate_integers(3.3, 9.9) == [4, 6, 8]


def test_generate_integers_conditional_branches():
    assert generate_integers(1, 1) == []
    assert generate_integers(1, 2) == []
    assert generate_integers(8, 8) == []
    assert generate_integers(9, 9) == []


def test_generate_integers_loop_iterations():
    assert generate_integers(1, 1) == []
    assert generate_integers(1, 1) == []
    assert generate_integers(1, 1) == []
    assert generate_integers(1, 1) == []


def test_generate_integers_error_cases():
    with pytest.raises(TypeError):
        generate_integers("5", 7)
    with pytest.raises(TypeError):
        generate_integers(5, "7")
    with pytest.raises(TypeError):
        generate_integers("5", "7")