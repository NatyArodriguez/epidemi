""" travers is a function to agroup the columns of a matrix by ten"""

import numpy as np


def travers_column(array):
    """With the max by columns, obtain mean and standar desviation.

    Args:
        array (ndarray) : Array with your data

    Returns:
        [shapes, mean, std] (list)
    """
    
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    
    if array.shape[1] % 10 != 0:
        raise ValueError("Number of columns must be a multiple of 10.")
    
    rango = array.shape[1]
    step = rango//10
    
    max_values_column = np.empty(step)
    std_values_column = np.empty(step)
    shapes = np.arange(10, rango+1, 10)
    
    for i in range(step):
        end_col = (i + 1) * 10
        matriz= array[:, :end_col]
        max_columns = np.max(matriz,axis=0)
        x = np.mean(max_columns)
        y = np.std(max_columns)/np.sqrt(shapes[i])
        
        max_values_column[i] = x
        std_values_column[i] = y.item()
        
    return [shapes, max_values_column, std_values_column]


def travers_mean(array):
    """With the max of the average of the columns, obtain mean and standar\
        desviation.

    Args:
        array (ndarray) : Array with your data

    Returns:
        [shapes, mean, std] (list)
    """
    
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    
    if array.shape[1] % 10 != 0:
        raise ValueError("Numer of columns must be a multiple of 10.")
    
    rango = array.shape[1]
    step = rango//10
    
    max_values_mean = np.empty(step)
    std_values_mean = np.empty(step)
    shapes = np.arange(10, rango+1 , 10)
    
    for i in range(step):
        end_col = (i + 1) * 10
        matriz= array[:, :end_col]

        u = np.mean(matriz, axis=1)
        v = np.std(matriz, axis =1)
        
    
        index_max = np.argmax(u)
        p_max = u[index_max]
        std_max = v[index_max]/np.sqrt(shapes[i])

        max_values_mean[i] = p_max
        std_values_mean[i] = std_max
    return [shapes, max_values_mean, std_values_mean]


def recorrer(array):
    rango = array.shape[1]
    step = rango//10
    
    max_values_column = np.empty(step)
    max_values_mean = np.empty(step)
    std_values_column = np.empty(step)
    std_values_mean = np.empty(step)
    shapes = np.arange(10, rango+1, 10)
    
    for i in range(step):
        if i < rango:
            end_col = (i + 1) * 10
            matriz= array[:,:end_col]
            max_columns = np.max(matriz,axis=0)
            x = np.mean(max_columns)
            y = np.std(max_columns)/np.sqrt(shapes[i])

            u = np.mean(matriz, axis=1)
            v = np.std(matriz, axis =1)
            
            p_max = u.max()
            where_max = np.where(u == p_max)
            std_max = v[where_max]/np.sqrt(shapes[i])
            
            max_values_column[i] = x
            std_values_column[i] = y
            max_values_mean[i] = p_max
            std_values_mean[i] = std_max
    return shapes, max_values_column, std_values_column, max_values_mean, std_values_mean

