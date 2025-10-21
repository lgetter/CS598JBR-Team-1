import pytest
from HumanEval_89 import encrypt  # replace 'HumanEval_89' with the name of the module where the function is defined


def test_encrypt():
    assert encrypt('hello') == 'jgnnq'
    assert encrypt('HELLO') == 'HELLO'  # uppercase letters remain unchanged
    assert encrypt('WORLD') == 'YQTWRF'  # uppercase letters are shifted
    assert encrypt('') == ''  # empty string remains empty
    assert encrypt('a') == 'c'  # single character is shifted
    assert encrypt('z') == 'b'  # 'z' is shifted to 'b'
    assert encrypt('Z') == 'B'  # 'Z' is shifted to 'B'
    assert encrypt('123') == '123'  # numbers and special characters remain unchanged
    assert encrypt(' ') == ' '  # space remains a space
    assert encrypt('abcdefghijklmnopqrstuvwxyz') == 'cdefghijklmnopqrstuvwxyzab'  # whole alphabet is shifted


def test_encrypt_errors():
    with pytest.raises(TypeError):
        encrypt(123)  # input must be a string
    with pytest.raises(TypeError):
        encrypt(None)  # input must be a string