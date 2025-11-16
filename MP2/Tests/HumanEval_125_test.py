import pytest
from HumanEval_125 import split_words  # replace with your actual module name

def test_split_words_with_spaces():
    assert split_words("Hello world") == ["Hello", "world"]

def test_split_words_with_commas():
    assert split_words("Hello,world") == ["Hello", "world"]

def test_split_words_without_spaces_or_commas():
    assert split_words("HelloWorld") == 5  # 5 is the count of lowercase even ascii characters

def test_split_words_empty_string():
    assert split_words("") == []

def test_split_words_single_word():
    assert split_words("Hello") == 5

def test_split_words_max_values():
    assert split_words("a"*10000) == 10000

def test_split_words_boundary_conditions():
    assert split_words("a"*1000000) == 1000000

def test_split_words_different_data_types():
    with pytest.raises(TypeError):
        split_words(123)

def test_split_words_if_else_statements():
    assert split_words("Hello,world") == ["Hello", "world"]
    assert split_words("Hello world") == ["Hello", "world"]
    assert split_words("HelloWorld") == 5

def test_split_words_loop_iterations():
    assert split_words("") == []
    assert split_words("Hello") == 5
    assert split_words("Hello World") == ["Hello", "World"]

def test_split_words_error_cases():
    with pytest.raises(TypeError):
        split_words(None)
    with pytest.raises(TypeError):
        split_words(["Hello", "world"])