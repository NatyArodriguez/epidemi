import pytest
import numpy as np


from epidemi.core.recorrer import (
    travers_column,
    travers_mean
)


def test_travers_column_valid_input():
    array = np.random.rand(5, 20)
    shapes, mean, std = travers_column(array)
    assert len(shapes) == 2
    assert mean.shape == (2,)
    assert std.shape == (2,)


def test_travers_mean_valid_input():
    array = np.random.rand(5, 20)
    shapes, mean, std = travers_mean(array)
    assert len(shapes) == 2
    assert mean.shape == (2,)
    assert std.shape == (2,)

def test_travers_column_wrong_input():
    with pytest.raises(TypeError):
        travers_column([1,2,3],[4,5,6])

def test_travers_mean_wrong_input():
    with pytest.raises(TypeError):
        travers_mean([1,2,3],[4,5,6])

def test_travers_column_wrong_columns():
    array = np.random.rand(5,8)
    with pytest.raises(ValueError):
        travers_column(array)

def test_travers_mean_wrong_columns():
    array = np.random.rand(5,8)
    with pytest.raises(ValueError):
        travers_mean(array)