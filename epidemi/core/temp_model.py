""" Functions for temperature model"""
# Tal vez deba agregar las funciones para calcular la media y desviacion por decil
import numpy as np

def anual_temp(data_t, data_std, data_days):
    temp_array = np.empty([0])
    long = len(data_t)
    for i in range(0,long):
        centro = data_t[i]
        std = data_std[i]
        days = data_days[i]
        aux = np.random.normal(centro, std, days)
        temp_array = np.concatenate([temp_array, aux])
    return temp_array

