import pytest
from HumanEval_30 import get_positive  # replace with your actual module name


def test_get_positive():
    # Normal/typical inputs
    assert get_positive([1, 2, 3, -4, 5]) == [1, 2, 3, 5]
    assert get_positive([10, 20, 30, -40, 50]) == [10, 20, 30, 50]

    # Edge cases (empty inputs, single elements, maximum values)
    assert get_positive([]) == []
    assert get_positive([1]) == [1]
    assert get_positive([-1]) == []
    assert get_positive([1000000]) == [1000000]
    assert get_positive([-1000000]) == []

    # Boundary conditions
    assert get_positive([0]) == []
    assert get_positive([-1, -2, -3, 0, 1, 2, 3]) == [1, 2, 3]

    # Different data types where applicable
    assert get_positive([1.0, 2.0, 3.0, -4.0, 5.0]) == [1.0, 2.0, 3.0, 5.0]
    assert get_positive(["1", "2", "3", "-4", "5"]) == ["1", "2", "3", "5"]

    # All conditional branches (if/else statements)
    assert get_positive([1, 2, 3, -4, 5]) == [1, 2, 3, 5]
    assert get_positive([-1, -2, -3, -4, -5]) == []

    # Loop iterations (empty, single, multiple)
    assert get_positive([]) == []
    assert get_positive([1]) == [1]
    assert get_positive([-1, 1]) == [1]

    # Error cases and exceptions
    with pytest.raises(TypeError):
        get_positive(None)
    with pytest.raises(TypeError):
        get_positive("1,2,3")