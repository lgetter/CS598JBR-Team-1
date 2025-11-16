import pytest
from HumanEval_147 import get_max_triples  # replace 'HumanEval_147' with the name of the module where the function is defined

def test_get_max_triples():
    # Normal/typical inputs
    assert get_max_triples(5) == 1
    assert get_max_triples(10) == 4

    # Edge cases (empty inputs, single elements, maximum values)
    with pytest.raises(IndexError):  # n is less than 1
        get_max_triples(0)
    assert get_max_triples(1) == 0  # single element
    assert get_max_triples(1000) > 0  # maximum value

    # Boundary conditions
    assert get_max_triples(1) == 0  # single element
    assert get_max_triples(2) == 0  # two elements
    assert get_max_triples(3) == 1  # three elements

    # Different data types where applicable
    with pytest.raises(TypeError):  # n is not an integer
        get_max_triples('5')
    with pytest.raises(TypeError):  # n is a float
        get_max_triples(5.0)

    # All conditional branches (if/else statements)
    assert get_max_triples(2) == 0  # n is less than 3
    assert get_max_triples(3) == 1  # n is 3

    # Loop iterations (empty, single, multiple)
    assert get_max_triples(0) == 0  # empty
    assert get_max_triples(1) == 0  # single element
    assert get_max_triples(2) == 0  # two elements

    # Error cases and exceptions
    with pytest.raises(IndexError):  # n is less than 1
        get_max_triples(0)
    with pytest.raises(TypeError):  # n is not an integer
        get_max_triples('5')
    with pytest.raises(TypeError):  # n is a float
        get_max_triples(5.0)