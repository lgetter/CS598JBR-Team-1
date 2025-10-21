import pytest
from HumanEval_11 import string_xor  # replace with your actual module name


def test_string_xor():
    assert string_xor('000', '001') == '001'
    assert string_xor('111', '101') == '010'
    assert string_xor('1111', '1010') == '0101'
    assert string_xor('', '') == ''
    assert string_xor('1', '1') == '0'
    assert string_xor('0', '1') == '1'
    assert string_xor('11111', '11111') == '00000'
    assert string_xor('10101010', '01010101') == '11111111'

    with pytest.raises(TypeError):
        string_xor(123, '1010')

    with pytest.raises(TypeError):
        string_xor('1010', 'abc')

    with pytest.raises(ValueError):
        string_xor('1010', '10101')

    with pytest.raises(ValueError):
        string_xor('10101', '1010')