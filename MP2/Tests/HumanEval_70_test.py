import pytest
from HumanEval_70 import strange_sort_list  # replace with your module name


def test_strange_sort_list():
    # Normal/typical inputs
    assert strange_sort_list([1, 2, 3, 4, 5]) == [5, 1, 4, 2, 3]
    assert strange_sort_list([5, 4, 3, 2, 1]) == [5, 1, 4, 2, 3]
    assert strange_sort_list([1, 3, 2, 5, 4]) == [5, 1, 4, 2, 3]

    # Edge cases (empty inputs, single elements, maximum values)
    assert strange_sort_list([]) == []
    assert strange_sort_list([1]) == [1]
    assert strange_sort_list([100]) == [100]

    # Boundary conditions
    assert strange_sort_list([1, 1]) == [1, 1]
    assert strange_sort_list([1, 2, 2]) == [2, 1, 2]

    # Different data types where applicable
    assert strange_sort_list([1, 2, 3, 4, '5']) == [5, 1, 4, 2, 3]
    assert strange_sort_list(['5', '4', '3', '2', '1']) == ['5', '1', '4', '2', '3']

    # All conditional branches (if/else statements)
    assert strange_sort_list([1, 2, 3, 4, 5]) == [5, 1, 4, 2, 3]
    assert strange_sort_list([]) == []

    # Loop iterations (empty, single, multiple)
    assert strange_sort_list([]) == []
    assert strange_sort_list([1]) == [1]
    assert strange_sort_list([1, 2, 3, 4, 5]) == [5, 1, 4, 2, 3]

    # Error cases and exceptions
    with pytest.raises(TypeError):
        strange_sort_list(None)
    with pytest.raises(TypeError):
        strange_sort_list({1, 2, 3})