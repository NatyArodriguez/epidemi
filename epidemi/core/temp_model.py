"""Diferents models to generate daily serie of temperature"""
# Tal vez deba agregar las funciones para calcular la media y desviacion por decil
import numpy as np

from scipy.stats import gamma

def anual_temp(mean_params, std_params,
                                shape_params, scale_params, days_decil):
    """Random values of temperature are obtained\
        following the normal distribution of each\
        decile chatacterized by its mean and\
        standard deviation.

    Args:
        data_t (np_array): average temperature per decil.
        data_std (np_array): std per decil.
        data_days (np_array): numbers of day per decil

    Returns:
        np_array: A daily serie of temperature
    """
    
    total_days = np.sum(days_decil)
    
    max_temp_limit = 46
    min_temp_limit = -2
    
    intento = 0
    max_intentos = 1000
    
    while intento < max_intentos:
        tmean_array = np.empty(total_days)
        tmax_array = np.empty(total_days)
        current_index = 0
        limites_excedidos = False
        
        for i, days in enumerate(days_decil):
            daily_means = np.random.normal(mean_params[i], std_params[i], days)
            variations = gamma.rvs(a=shape_params[i], scale=scale_params[i], 
                                  loc=0, size=days)
            daily_maxs = daily_means + variations
            daily_mins = 2 * daily_means - daily_maxs
            
            if np.any(daily_maxs > max_temp_limit)or np.any(daily_mins < min_temp_limit):
                limites_excedidos = True
                break
            
            end_index = current_index + days
            tmean_array[current_index:end_index] = daily_means
            tmax_array[current_index:end_index] = daily_maxs
            current_index = end_index
        
        if not limites_excedidos and current_index == total_days:
            tmin_array = 2 * tmean_array - tmax_array
            return tmin_array, tmax_array, tmean_array
        
        intento += 1

class TemperatureGenerator:
    """Generator of daily temperature serie as a result\
        of point interpolation.
    """
    
    # NUMBER OF DAYS PER MONTH IN A LEAP YEAR 
    MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    def __init__(self, max_temp_limit=46, min_temp_limit=-2, max_attempts=1000):
        "Initializes decil ranges and temperature limits"
        self.decil_ranges = self._generate_decil_ranges()
        self.max_temp_limit = max_temp_limit
        self.min_temp_limit = min_temp_limit
        self.max_attempts = max_attempts
    
    def _generate_decil_ranges(self):
        """Generates decile intervals (36 in total)
        """
        decil_ranges = []
        day_counter = 0
        decil = 1
        
        for days in self.MONTH_DAYS:
            if days == 31:
                group_sizes = [10, 10, 11]
            elif days == 30:
                group_sizes = [10, 10, 10]
            elif days == 28:
                group_sizes = [10, 10, 8]
        
            start = 0
            for group_size in group_sizes:
                end = start + group_size - 1
                decil_ranges.append((day_counter + start,
                                    day_counter + end, decil))
                start = end + 1
                decil += 1
            
            day_counter += days
        
        return decil_ranges
    
    @staticmethod
    def _linear_interpolation(p1, p2):
        """Determines a straight line between two points.

        Args:
            pi = (day, temperature)
        """
        x1, y1 = p1
        x2, y2 = p2
        
        if x2 == x1:
            return 0, y1
        
        m = (y2-y1) / (x2-x1)
        b = y1 - m * x1
        return m, b
    
    def get_decil(self, day):
        """find the decile corresponding to a specific day

        Args:
            day (_type_): _description_
        """
        for start, end, decil in self.decil_ranges:
            if start <= day <= end:
                return decil
    
    def generate_daily_temperature(self, points):
        """Generates a daily temperature series by linear interpolation.

        Args:
            points (np_array): Array of points [[days, temperature], ...]
        
        Returns:
            Array with daily temperatures
        """
        if len(points) < 2:
            raise ValueError("Must be at least two points")
    
        daily_temp = []
        
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i+1]
            x1, x2 = p1[0], p2[0]
            
            m, b = self._linear_interpolation(p1,p2)
            
            n_points = int((x2-x1))
            xs = np.linspace(x1, x2, n_points, endpoint=False)
            ys = m * xs + b
            daily_temp.append(ys)
        daily_temp.append([points[-1][1]])
        
        return np.concatenate(daily_temp)
    
    def _validate_temperature_limits(self, max_value, min_value):
        """Valida que las temperaturas estén dentro de los límites permitidos.
        
        Args:
            max_value (float): Temperatura máxima
            min_value (float): Temperatura mínima
            
        Returns:
            bool: True si están dentro de los límites
        """
        return (max_value <= self.max_temp_limit and 
                min_value >= self.min_temp_limit)
    
    def generate_temperature_points(self, days_interval, mean_params, std_params,
                                    shape_params, scale_params):
        """Generates temperature points for specific days according\
            to the decile they belong to.
        Args:
            days_interval (_type_): Array of days.
            mean_params (_type_): Data of mean temperature per decil.
            std_params (_type_): Data of standard deviation per decil.
            shape_params (_type_): Shape parameters for gamma distribution.
            scale_params (_type_): Scale parameters for gamma distribution.
        Returns:
            Array of points [[days, temperature], ...]
        """
        p_tmax = []
        p_tmean = []
        p_tmin = []
        
        for day in days_interval:
            decil = self.get_decil(day)
            # First decil correspond to 0 index
            idx = decil - 1
            
            # Generar temperaturas con validación de límites
            attempt = 0
            valid_temperature = False
            
            while not valid_temperature and attempt < self.max_attempts:
                # Generar valores de temperatura
                mean_value = np.random.normal(mean_params[idx], std_params[idx])
                variations = gamma.rvs(a=shape_params[idx], loc=0,
                                       scale=scale_params[idx])
                max_value = mean_value + variations
                min_value = mean_value - variations
                
                # Validar límites
                if self._validate_temperature_limits(max_value, min_value):
                    valid_temperature = True
                else:
                    attempt += 1
            
            p_tmean.append([day, mean_value])
            p_tmax.append([day, max_value])
            p_tmin.append([day, min_value])
        
        return np.array(p_tmin), np.array(p_tmax), np.array(p_tmean)

class TemperatureGeneratorWithNoise:
    """Generator of daily temperature serie as a result\
        of point interpolation.
    """
    
    # NUMBER OF DAYS PER MONTH IN A LEAP YEAR 
    MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    def __init__(self):
        "Initializes decil ranges"
        self.decil_ranges = self._generate_decil_ranges()
    
    def _generate_decil_ranges(self):
        """Generates decile intervals (36 in total)
        """
        decil_ranges = []
        day_counter = 0
        decil = 1
        
        for days in self.MONTH_DAYS:
            if days == 31:
                group_sizes = [10, 10, 11]
            elif days == 30:
                group_sizes = [10, 10, 10]
            elif days == 28:
                group_sizes = [10, 10, 8]
        
            start = 0
            for group_size in group_sizes:
                end = start + group_size - 1
                decil_ranges.append((day_counter + start,
                                    day_counter + end, decil))
                start = end + 1
                decil += 1
            
            day_counter += days
        
        return decil_ranges
    
    @staticmethod
    def _linear_interpolation(p1, p2):
        """Determines a straight line between two points.

        Args:
            pi = (day, temperature)
        """
        x1, y1 = p1
        x2, y2 = p2
        
        if x2 == x1:
            return 0, y1
        
        m = (y2-y1) / (x2-x1)
        b = y1 - m * x1
        return m, b
    
    def get_decil(self, day):
        """find the decile corresponding to a specific day

        Args:
            day (_type_): _description_
        """
        for start, end, decil in self.decil_ranges:
            if start <= day <= end:
                return decil
    
    def generate_daily_temperature_with_noise(self, points, std_params, std_max, std_min,
                                              temp_type, 
                                              reference_series=None):
        """Generates a daily temperature series by linear interpolation with noise applied only to intermediate points.

        Args:
            points (np_array): Array of points [[days, temperature], ...]
            std_params: Standard deviation parameters for normal noise
            std_max: Standard deviation for maximum temperature
            std_min: Standard deviation for minimum temperature
            temp_type: 'mean', 'max', or 'min'
            reference_series: Series de referencia para verificar condiciones (tmean para tmax/tmin)
        
        Returns:
            Array with daily temperatures
        """
        if len(points) < 2:
            raise ValueError("Must be at least two points")
    
        daily_temp = []
        
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i+1]
            x1, x2 = p1[0], p2[0]
            
            m, b = self._linear_interpolation(p1, p2)
            
            n_points = int((x2 - x1))
            xs = np.linspace(x1, x2, n_points, endpoint=False)
            ys = m * xs + b
            
            noisy_ys = []
            for j, (x, y) in enumerate(zip(xs, ys)):
                day = int(round(x))
                decil = self.get_decil(day)
                idx = decil - 1
                
                # Solo aplicar ruido si no es un punto original
                is_original_point = (x == x1 and j == 0) or (x == x2 and j == len(xs)-1)
                
                if not is_original_point:
                    if temp_type == 'mean':
                        noise = np.random.normal(0, std_params[idx])
                        noisy_ys.append(y + noise)
                    
                    elif temp_type == 'max':
                        max_attempts = 1000
                        for attempt in range(max_attempts):
                            noise = np.random.normal(0, std_max[idx])
                            temp_candidate = y + noise
                            
                            # Verificar que tmax > tmean
                            if reference_series is not None and day < len(reference_series):
                                if temp_candidate > reference_series[day]:
                                    noisy_ys.append(temp_candidate)
                                    break
                            else:
                                # Si no hay serie de referencia, aceptar cualquier valor
                                noisy_ys.append(temp_candidate)
                                break
                            
                            # Si llegamos al último intento, usar el valor aunque no cumpla
                            if attempt == max_attempts - 1:
                                noisy_ys.append(max(temp_candidate, reference_series[day] + 1.0))
                    
                    elif temp_type == 'min':
                        max_attempts = 1000
                        for attempt in range(max_attempts):
                            noise = np.random.normal(0, std_min[idx])
                            temp_candidate = y + noise
                            
                            # Verificar que tmin < tmean
                            if reference_series is not None and day < len(reference_series):
                                if temp_candidate < reference_series[day]:
                                    noisy_ys.append(temp_candidate)
                                    break
                            else:
                                noisy_ys.append(temp_candidate)
                                break
                            
                            if attempt == max_attempts - 1:
                                noisy_ys.append(min(temp_candidate, reference_series[day] - 1.0))
                else:
                    noisy_ys.append(y)
            
            daily_temp.append(noisy_ys)
        
        daily_temp.append([points[-1][1]])
        
        return np.concatenate(daily_temp)
    
    def generate_temperature_points(self, days_interval, mean_params, std_params, std_max, std_min,
                                    shape_params, scale_params):
        """Generates temperature points for specific days according\
            to the decil they belong to.
            
        Args:
            days_interval (_type_): Array of days.
            mean_params (_type_): Mean temperature parameters per decil.
            std_params (_type_): Standard deviation parameters per decil.
            std_max: Standard deviation for maximum temperature per decil
            std_min: Standard deviation for minimum temperature per decil
            shape_params (_type_): Shape parameters for gamma distribution.
            scale_params (_type_): Scale parameters for gamma distribution.
            
        Returns:
            Arrays of points and series
        """
        # Paso 1: Generar puntos iniciales para los días del intervalo
        p_tmean_initial = []
        p_tmax_initial = []
        p_tmin_initial = []
        
        for day in days_interval:
            decil = self.get_decil(day)
            idx = decil - 1
            
            mean_value = np.random.normal(mean_params[idx], std_params[idx])
            p_tmean_initial.append([day, mean_value])
            
            gamma_variation = gamma.rvs(a=shape_params[idx], loc=0, scale=scale_params[idx])
            
            max_value = mean_value + gamma_variation
            min_value = mean_value - gamma_variation
            
            p_tmax_initial.append([day, max_value])
            p_tmin_initial.append([day, min_value])
        
        t_tmean_final = self.generate_daily_temperature_with_noise(
            p_tmean_initial, std_params, std_max, std_min, 'mean')
        
        t_tmax_final = self.generate_daily_temperature_with_noise(
            p_tmax_initial, std_params, std_max, std_min,
            'max', t_tmean_final)
        
        t_tmin_final = self.generate_daily_temperature_with_noise(
            p_tmin_initial, std_params, std_max, std_min,
            'min', t_tmean_final)
        
        return t_tmin_final, t_tmax_final, t_tmean_final