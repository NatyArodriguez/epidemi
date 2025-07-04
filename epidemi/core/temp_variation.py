import numpy as np
import epidemi.core.utils as u
from .temp_model import anual_temp
from .utils import load_data_file


# decil | days | tmin | tmax | tmean | std tmin | std tmax | std tmean 
temp = load_data_file('hystoric_temp.txt')

# tmin	| tmax	| tmean | tmedian | rain | hrm | hr
oran_medio = load_data_file('Oran_2001_2017_medio.txt')

fechas_oran = np.arange('2001-01-01', '2018-01-01', dtype='datetime64[D]')
fechas_str = fechas_oran.astype(str) 
mask_no_29feb = np.array([not f.endswith('02-29') for f in fechas_str])
fechas_oran = fechas_oran[mask_no_29feb]
oran_medio = oran_medio[mask_no_29feb]

rain = oran_medio[:,4]
hr = oran_medio[:,6]

def temp_matrix_epic_size(delta, n_iterations, season, suma, data_T=temp):
    """This function generates a matrix where each column represents the daily\
        total epidemic size for a year, simulated under different rainfall\
        scenaries specific by the tuple sigmas.

    Parameters:
        sigmas: tuple of float
            tuple with two entrance, the firts it most be sigma rain and\
            the second must be sigma alpha.
        n_iterations: int
            Number of iteration that you wish.
    """
    date_i = np.datetime64(suma[0])
    date_f = np.datetime64(suma[1])
    ci_sim = np.arange(date_i, date_f + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    ci_sim_str = ci_sim.astype(str)
    mask_no_29feb = np.array([not f.endswith('02-29') for f in ci_sim_str])

    ci_sim = ci_sim[mask_no_29feb]
    
    n_sim = len(ci_sim)
    
    matriz = np.zeros([n_sim, n_iterations])
    
    std_tmin = data_T[:,5]
    std_tmean = data_T[:,7]
    
    tmin_syntetic = data_T[:,2] + delta
    tmean_syntetic = data_T[:,4] + delta
    
    days_decil = data_T[:,1].astype(int)

    for i in range(n_sim):
        ci = [ci_sim[i], 1]
        for j in range(n_iterations):
            tmin_serie = anual_temp(tmin_syntetic, std_tmin, days_decil)
            tmean_serie = anual_temp(tmean_syntetic, std_tmean, days_decil)
            
            tmin = np.tile(tmin_serie, 17)
            tmean = np.tile(tmean_serie, 17)
            aux = u.fun(350, 1.69, season, suma, ci, rain, tmin,
                        tmean, hr)
            matriz[i,j] = aux[1]
    return matriz

def temp_matrix_epic_size2(sigmas, n_iterations, season, suma, data_T=temp):
    """This function generates a matrix where each column represents the daily\
        total epidemic size for a year, simulated under different rainfall\
        scenaries specific by the tuple sigmas.

    Parameters:
        sigmas: tuple of float
            tuple with two entrance, the firts it most be sigma rain and\
            the second must be sigma alpha.
        n_iterations: int
            Number of iteration that you wish.
    """
    date_i = np.datetime64(suma[0])
    date_f = np.datetime64(suma[1])
    ci_sim = np.arange(date_i, date_f + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    ci_sim_str = ci_sim.astype(str)
    mask_no_29feb = np.array([not f.endswith('02-29') for f in ci_sim_str])

    ci_sim = ci_sim[mask_no_29feb]
    
    n_sim = len(ci_sim)
    
    matriz = np.zeros([n_sim, n_iterations])
    sigma_tmin = sigmas[0]
    sigma_tmean = sigmas[1]
    
    std_tmin = data_T[:,5]
    std_tmax = data_T[:,6]
    
    tmin_syntetic = std_tmin*sigma_tmin + data_T[:,2]
    tmax_syntetic = std_tmax*sigma_tmean + data_T[:,3]
    
    days_decil = data_T[:,1].astype(int)

    for i in range(n_sim):
        ci = [ci_sim[i], 1]
        for j in range(n_iterations):
            tmin_serie = anual_temp(tmin_syntetic, std_tmin, days_decil)
            tmax_serie = anual_temp(tmax_syntetic, std_tmax, days_decil)
            
            tmean_serie = (tmin_serie + tmax_serie)/2
            
            tmin = np.tile(tmin_serie, 17)
            tmean = np.tile(tmean_serie, 17)
            aux = u.fun(350, 1.69, season, suma, ci, rain, tmin,
                        tmean, hr)
            matriz[i,j] = aux[1]
    return matriz
