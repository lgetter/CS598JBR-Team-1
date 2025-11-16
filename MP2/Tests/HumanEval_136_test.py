import pytest
from HumanEval_136 import largest_smallest_integers  # replace with your actual module


def test_largest_smallest_integers():
    # Normal/typical inputs
    assert largest_smallest_integers([1, 2, 3, -4, -5]) == (-5, 3)
    assert largest_smallest_integers([0, 1, 2, -3, -4]) == (-3, 2)
    assert largest_smallest_integers([-1, -2, -3, -4, -5]) == (-1, None)
    assert largest_smallest_integers([1, 2, 3, 4, 5]) == (None, 5)

    # Edge cases (empty inputs, single elements, maximum values)
    assert largest_smallest_integers([]) == (None, None)
    assert largest_smallest_integers([1]) == (None, 1)
    assert largest_smallest_integers([-1]) == (-1, None)
    assert largest_smallest_integers([10**6]) == (None, 10**6)

    # Boundary conditions
    assert largest_smallest_integers([-10**6, 10**6]) == (-10**6, 10**6)
    assert largest_smallest_integers([-10**6, 0, 10**6]) == (-10**6, 10**6)

    # Different data types where applicable
    assert largest_smallest_integers([1, 2.0, -3, -4.0]) == (-4.0, 2.0)

    # All conditional branches (if/else statements)
    assert largest_smallest_integers([-1, 0, 1]) == (0, 1)
    assert largest_smallest_integers([-1, -2, -3]) == (-1, None)
    assert largest_smallest_integers([1, 2, 3]) == (None, 3)

    # Loop iterations (empty, single, multiple)
    assert largest_smallest_integers([-1, -1, -1]) == (-1, None)
    assert largest_smallest_integers([1, 1, 1]) == (None, 1)

    # Error cases and exceptions
    with pytest.raises(TypeError):
        largest_smallest_integers(["1", "2", "3"])
    with pytest.raises(TypeError):
        largest_smallest_integers([[1], [2], [3]])