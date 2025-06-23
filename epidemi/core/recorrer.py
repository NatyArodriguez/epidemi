import numpy as np


def recorrer(array):
    rango = array.shape[1]
    step = rango//10
    
    max_values_column = np.empty(step)
    max_values_mean = np.empty(step)
    std_values_column = np.empty(step)
    std_values_mean = np.empty(step)
    shapes = np.arange(10, rango+10, 10)
    
    for i in range(step):
        if i < rango:
            matriz= array[:,0:i+10]
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

