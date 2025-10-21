import pytest
from HumanEval_81 import numerical_letter_grade  # replace with your module name


def test_numerical_letter_grade():
    assert numerical_letter_grade([4.0]) == ["A+"]
    assert numerical_letter_grade([3.8]) == ["A"]
    assert numerical_letter_grade([3.5]) == ["A-"]
    assert numerical_letter_grade([3.0]) == ["B+"]
    assert numerical_letter_grade([2.8]) == ["B"]
    assert numerical_letter_grade([2.5]) == ["B-"]
    assert numerical_letter_grade([2.0]) == ["C+"]
    assert numerical_letter_grade([1.8]) == ["C"]
    assert numerical_letter_grade([1.5]) == ["C-"]
    assert numerical_letter_grade([1.0]) == ["D+"]
    assert numerical_letter_grade([0.8]) == ["D"]
    assert numerical_letter_grade([0.5]) == ["D-"]
    assert numerical_letter_grade([0.0]) == ["E"]
    assert numerical_letter_grade([4.5]) == ["A+"]
    assert numerical_letter_grade([3.7, 3.9]) == ["A", "A+"]
    assert numerical_letter_grade([2.2, 2.6]) == ["B", "B+"]
    assert numerical_letter_grade([1.2, 1.6]) == ["C", "C+"]
    assert numerical_letter_grade([0.2, 0.6]) == ["D", "D+"]
    assert numerical_letter_grade([4.0, 3.0, 2.0, 1.0, 0.0]) == ["A+", "B+", "C+", "D+", "E"]
    assert numerical_letter_grade([]) == []


def test_numerical_letter_grade_errors():
    with pytest.raises(TypeError):
        numerical_letter_grade(None)
    with pytest.raises(TypeError):
        numerical_letter_grade("abc")
    with pytest.raises(TypeError):
        numerical_letter_grade({4.0})
    with pytest.raises(TypeError):
        numerical_letter_grade([4.0, "abc"])
    with pytest.raises(ValueError):
        numerical_letter_grade([4.0, 5.0])
    with pytest.raises(ValueError):
        numerical_letter_grade([-1.0])