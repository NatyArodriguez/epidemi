import numpy as np

from scipy.stats import gamma

class TemperatureGeneratorWithNoiseStep:
    """Generator of daily temperature serie as a result\
        of point interpolation.
    """
    
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
        """Find the decile corresponding to a specific day
        """
        for start, end, decil in self.decil_ranges:
            if start <= day <= end:
                return decil
        # Si el día está fuera del rango (365), usar el último decil
        return 36
    
    def generate_spaced_days(self, spacing):
        """Generate an array of days spaced to cover exactly 365 days.
        
        Args:
            spacing : Gap between days (2, 3, 5, etc.).
            
        Returns:
            Days from 0 to 364 with the specified spacing.
        """
        # Calcular cuántos puntos necesitamos para cubrir 365 días
        n_points = int(np.ceil(365 / spacing))
        
        # Generar días desde 0 hasta el último punto que no exceda 364
        days = np.arange(0, n_points * spacing, spacing)
        
        # Asegurar que el último día sea 364 (día 365 sería índice 364)
        if days[-1] > 364:
            days = days[days <= 364]
        
        # Si el último día no es 364, agregarlo
        if days[-1] != 364:
            days = np.append(days, 364)
        
        return days
    
    def _validate_temperature_limits(self, max_value, min_value):
        """Check that the temperatures are within the permitted limits.
        
        Args:
            max_value : Maximum temperature.
            min_value : Minimum temperature.
            
        Returns:
            bool: True if they are within the permitted limits.
        """
        return (max_value <= self.max_temp_limit and 
                min_value >= self.min_temp_limit)
    
    def _apply_temperature_limits(self, temperature, temp_type=None):
        """Apply temperature limits to a single value.
        
        Args:
            temperature : Temperature at which the limit applies.
            temp_type : 'max', 'min', or None for both limits.
            
        Returns:
            float: Temperature within the limits.
        """
        if temp_type == 'max':
            return min(temperature, self.max_temp_limit)
        elif temp_type == 'min':
            return max(temperature, self.min_temp_limit)
        else:
            return np.clip(temperature, self.min_temp_limit, self.max_temp_limit)
    
    def generate_daily_temperature_with_noise(self, points, std_params, std_max, std_min,
                                              temp_type, reference_series=None):
        """Generates a daily temperature series by linear interpolation with noise applied only to intermediate points.

        Args:
            points : Array of points [[days, temperature], ...]
            std_params: Standard deviation parameters per decade.
            std_max: Standard deviation for maximum temperature per decade.
            std_min: Standard deviation for minimum temperature per decade.
            shape_params: Shape parameters for gamma distribution.
            scale_params: Scale parameters for gamma distribution.
            temp_type: 'mean', 'max', or 'min'
            reference_series: Reference series for checking conditions (tmean for tmax/tmin)
        
        Returns:
            Array with daily temperatures (365 days).
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
            
            # Aplicar ruido solo a los puntos intermedios (no a los puntos originales)
            noisy_ys = []
            for j, (x, y) in enumerate(zip(xs, ys)):
                day = int(round(x))
                decil = self.get_decil(day)
                idx = decil - 1
                
                # Solo aplicar ruido si no es un punto original
                is_original_point = (x == x1 and j == 0) or (x == x2 and j == len(xs)-1)
                
                if not is_original_point:
                    if temp_type == 'mean':
                        # Ruido normal para temperatura media
                        noise = np.random.normal(0, std_params[idx])
                        temp_candidate = y + noise
                        # Aplicar límites a temperatura media
                        temp_candidate = self._apply_temperature_limits(temp_candidate)
                        noisy_ys.append(temp_candidate)
                    
                    elif temp_type == 'max':
                        # Ruido normal para temperatura máxima
                        for attempt in range(self.max_attempts):
                            noise = np.random.normal(0, std_max[idx])
                            temp_candidate = y + noise
                            
                            # Aplicar límite máximo
                            temp_candidate = self._apply_temperature_limits(temp_candidate, 'max')
                            
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
                            if attempt == self.max_attempts - 1:
                                if reference_series is not None and day < len(reference_series):
                                    enforced_temp = y + 0.5 #max(temp_candidate, reference_series[day] + 0.5) #0.1
                                    enforced_temp = self._apply_temperature_limits(enforced_temp, 'max')
                                    noisy_ys.append(enforced_temp)
                                else:
                                    temp_candidate = self._apply_temperature_limits(temp_candidate, 'max')
                                    noisy_ys.append(temp_candidate)
                    
                    elif temp_type == 'min':
                        # Ruido normal para temperatura mínima
                        for attempt in range(self.max_attempts):
                            noise = np.random.normal(0, std_min[idx])
                            temp_candidate = y + noise
                            
                            # Aplicar límite mínimo
                            temp_candidate = self._apply_temperature_limits(temp_candidate, 'min')
                            
                            # Verificar que tmin < tmean
                            if reference_series is not None and day < len(reference_series):
                                if temp_candidate < reference_series[day]:
                                    noisy_ys.append(temp_candidate)
                                    break
                            else:
                                # Si no hay serie de referencia, aceptar cualquier valor
                                noisy_ys.append(temp_candidate)
                                break
                            
                            # Si llegamos al último intento, usar el valor aunque no cumpla
                            if attempt == self.max_attempts - 1:
                                if reference_series is not None and day < len(reference_series):
                                    enforced_temp = y - 0.5 #min(temp_candidate, reference_series[day] - 0.5)
                                    enforced_temp = self._apply_temperature_limits(enforced_temp, 'min')
                                    noisy_ys.append(enforced_temp)
                                else:
                                    temp_candidate = self._apply_temperature_limits(temp_candidate, 'min')
                                    noisy_ys.append(temp_candidate)
                else:
                    # Mantener el punto original sin ruido, pero aplicar límites
                    original_temp = self._apply_temperature_limits(y, temp_type)
                    noisy_ys.append(original_temp)
            
            daily_temp.append(noisy_ys)
        
        # Agregar el último punto original (sin ruido, pero con límites aplicados)
        last_temp = self._apply_temperature_limits(points[-1][1], temp_type)
        daily_temp.append([last_temp])
        
        full_series = np.concatenate(daily_temp)
        
        # Asegurar que tenemos exactamente 365 días (índices 0-364)
        return full_series[:365]
    
    def _generate_temperature_point_with_limits(self, day, mean_params, std_params, 
                                               shape_params, scale_params):
        """Genera un punto de temperatura con validación de límites.
        
        Args:
            day : Day of the year.
            mean_params: Mean temperature parameters per decade.
            std_params: Standard deviation parameters per decade.
            shape_params : Shape parameters for gamma distribution.
            scale_params : Scale parameters for gamma distribution.
            
        Returns:
            tuple: (mean_value, max_value, min_value)
        """
        decil = self.get_decil(day)
        idx = decil - 1
        
        for attempt in range(self.max_attempts):
            # Generar valores de temperatura
            mean_value = np.random.normal(mean_params[idx], std_params[idx])
            variations = gamma.rvs(a=shape_params[idx], loc=0, scale=scale_params[idx])
            max_value = mean_value + variations
            min_value = mean_value - variations
            
            # Aplicar límites
            max_value = self._apply_temperature_limits(max_value, 'max')
            min_value = self._apply_temperature_limits(min_value, 'min')
            
            # Validar que max > min (consistencia física)
            if max_value > min_value:
                # Recalcular mean_value para mantener consistencia después de aplicar límites
                adjusted_mean = (max_value + min_value) / 2
                return adjusted_mean, max_value, min_value
        
        # Si no se encontraron valores válidos después de max_attempts
        # Usar valores por defecto dentro de los límites
        default_mean = (self.max_temp_limit + self.min_temp_limit) / 2
        default_max = self.max_temp_limit - 1
        default_min = self.min_temp_limit + 1
        
        return default_mean, default_max, default_min
    
    def generate_temperature_points(self, spacing, mean_params, std_params, std_max, std_min,
                                    shape_params, scale_params):
        """Generates temperature points for specific days according\
            to the decade they belong to.
            
        Args:
            spacing : gap between days (2, 3, 5, etc.)
            mean_params : Mean temperature parameters per decade.
            std_params : Standard deviation parameters per decade.
            std_max: Standard deviation for maximum temperature per decade.
            std_min: Standard deviation for minimum temperature per decade.
            shape_params : Shape parameters for gamma distribution.
            scale_params : Scale parameters for gamma distribution.
        Returns:
            Arrays with 365 points.
        """
        # Generar días espaciados
        days_interval = self.generate_spaced_days(spacing)
        
        # Paso 1: Generar puntos iniciales para los días del intervalo CON LÍMITES
        p_tmean_initial = []
        p_tmax_initial = []
        p_tmin_initial = []
        
        for day in days_interval:
            mean_value, max_value, min_value = self._generate_temperature_point_with_limits(
                day, mean_params, std_params, shape_params, scale_params)
            
            p_tmean_initial.append([day, mean_value])
            p_tmax_initial.append([day, max_value])
            p_tmin_initial.append([day, min_value])
        
        # Paso 2: Primero generar serie de tmean (sin dependencias)
        t_tmean_final = self.generate_daily_temperature_with_noise(
            p_tmean_initial, std_params, std_max, std_min, 'mean')
        
        # Paso 3: Generar tmax y tmin usando tmean como referencia
        t_tmax_final = self.generate_daily_temperature_with_noise(
            p_tmax_initial, std_params, std_max, std_min, 
            'max', t_tmean_final)
        
        t_tmin_final = self.generate_daily_temperature_with_noise(
            p_tmin_initial, std_params, std_max, std_min,
            'min', t_tmean_final)
        
        # Verificación final de límites
        final_max = np.max(t_tmax_final)
        final_min = np.min(t_tmin_final)
        
        if final_max > self.max_temp_limit or final_min < self.min_temp_limit:
            print(f"WARNING: Limits exceeded after interpolation.")
            print(f"Max: {final_max:.2f}°C, Limit: {self.max_temp_limit}°C")
            print(f"Min: {final_min:.2f}°C, Limit: {self.min_temp_limit}°C")
        
        return t_tmin_final, t_tmax_final, t_tmean_final

class TemperatureGeneratorWithNoiseStepLeap:
    """Generator of daily temperature series as a result of point interpolation.
    Supports both leap and non-leap years.
    """
    
    def __init__(self, leap_year=False, max_temp_limit=46, min_temp_limit=-2, max_attempts=1000):
        """Initializes decil ranges and temperature limits.
        
        Args:
            leap_year (bool): True for leap year (366 days), False for non-leap (365 days)
            max_temp_limit (float): Maximum temperature limit
            min_temp_limit (float): Minimum temperature limit
            max_attempts (int): Maximum attempts to find valid temperatures
        """
        self.leap_year = leap_year
        self.max_temp_limit = max_temp_limit
        self.min_temp_limit = min_temp_limit
        self.max_attempts = max_attempts
        
        # Set days per month based on leap year
        if leap_year:
            self.MONTH_DAYS = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            self.total_days = 366
        else:
            self.MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            self.total_days = 365
        
        self.decil_ranges = self._generate_decil_ranges()
    
    def _generate_decil_ranges(self):
        """Generates decile intervals (36 in total) with leap year adjustment.
        For February in leap years: 29 days distributed as 10, 10, 9.
        """
        decil_ranges = []
        day_counter = 0
        decil = 1
        
        for month_idx, days in enumerate(self.MONTH_DAYS):
            # Special handling for February in leap years
            if month_idx == 1 and self.leap_year:  # February (index 1)
                group_sizes = [10, 10, 9]  # 10, 10, 9 days for February in leap year
            elif days == 31:
                group_sizes = [10, 10, 11]
            elif days == 30:
                group_sizes = [10, 10, 10]
            elif days == 29:  # February in leap year
                group_sizes = [10, 10, 9]
            elif days == 28:  # February in non-leap year
                group_sizes = [10, 10, 8]
            else:
                # Fallback for any other cases
                base = days // 3
                remainder = days % 3
                group_sizes = [base + 1 if i < remainder else base for i in range(3)]
        
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
        
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b
    
    def get_decil(self, day):
        """Find the decile corresponding to a specific day.
        
        Args:
            day (int): Day of the year (0-based)
            
        Returns:
            int: Decile number (1-36)
        """
        for start, end, decil in self.decil_ranges:
            if start <= day <= end:
                return decil
        # If the day is out of range, use the last decil
        return 36
    
    def generate_spaced_days(self, spacing):
        """Genera un arreglo de días espaciados que cubre exactamente el año completo.
        
        Args:
            spacing (int): Espaciado entre días (2, 3, 5, etc.)
            
        Returns:
            np.array: Días desde 0 hasta total_days-1 con el espaciado especificado
        """
        # Calculate how many points we need to cover all days
        n_points = int(np.ceil(self.total_days / spacing))
        
        # Generate days from 0 to the last point not exceeding total_days-1
        days = np.arange(0, n_points * spacing, spacing)
        
        # Ensure the last day is total_days-1
        if days[-1] > self.total_days - 1:
            days = days[days <= self.total_days - 1]
        
        # If the last day is not total_days-1, add it
        if days[-1] != self.total_days - 1:
            days = np.append(days, self.total_days - 1)
        
        return days
    
    def _apply_temperature_limits(self, temperature, temp_type=None):
        """Aplica límites de temperatura a un valor individual.
        
        Args:
            temperature (float): Temperatura a limitar
            temp_type (str): 'max', 'min', o None para ambos límites
            
        Returns:
            float: Temperatura dentro de los límites
        """
        if temp_type == 'max':
            return min(temperature, self.max_temp_limit)
        elif temp_type == 'min':
            return max(temperature, self.min_temp_limit)
        else:
            return np.clip(temperature, self.min_temp_limit, self.max_temp_limit)
    
    def _generate_temperature_for_day(self, day, decil_idx, temp_type, 
                                     mean_params, std_params, std_max=None, std_min=None,
                                     shape_params=None, scale_params=None, 
                                     reference_value=None):
        """Genera temperatura para un día específico según el tipo de temperatura.
        
        Args:
            day: Día del año
            decil_idx: Índice del decil (0-based)
            temp_type: 'mean', 'max', o 'min'
            mean_params, std_params, std_max, std_min, shape_params, scale_params: Parámetros
            reference_value: Valor de referencia (para max/min basado en mean)
            
        Returns:
            float: Temperatura generada
        """
        for attempt in range(self.max_attempts):
            if temp_type == 'mean':
                temp = np.random.normal(mean_params[decil_idx], std_params[decil_idx])
                temp = self._apply_temperature_limits(temp)
                return temp
            
            elif temp_type == 'max':
                # Generar variación gamma positiva para tmax
                variation = gamma.rvs(a=shape_params[decil_idx], 
                                     loc=0, scale=scale_params[decil_idx])
                if reference_value is not None:
                    temp_candidate = reference_value + variation
                else:
                    # Si no hay referencia, usar mean como base
                    base_temp = np.random.normal(mean_params[decil_idx], std_params[decil_idx])
                    temp_candidate = base_temp + variation
                
                # Aplicar límite máximo y verificar consistencia
                temp_candidate = self._apply_temperature_limits(temp_candidate, 'max')
                
                if reference_value is None or temp_candidate > reference_value:
                    return temp_candidate
            
            elif temp_type == 'min':
                # Generar variación gamma positiva para tmin
                variation = gamma.rvs(a=shape_params[decil_idx], 
                                     loc=0, scale=scale_params[decil_idx])
                if reference_value is not None:
                    temp_candidate = reference_value - variation
                else:
                    # Si no hay referencia, usar mean como base
                    base_temp = np.random.normal(mean_params[decil_idx], std_params[decil_idx])
                    temp_candidate = base_temp - variation
                
                # Aplicar límite mínimo y verificar consistencia
                temp_candidate = self._apply_temperature_limits(temp_candidate, 'min')
                
                if reference_value is None or temp_candidate < reference_value:
                    return temp_candidate
        
        # Si no se encontró valor válido después de max_attempts
        if temp_type == 'mean':
            return np.clip(mean_params[decil_idx], self.min_temp_limit, self.max_temp_limit)
        elif temp_type == 'max':
            return self.max_temp_limit - 1
        elif temp_type == 'min':
            return self.min_temp_limit + 1
    
    def _add_february_leap_day_point(self, points, day, decil_idx, temp_type, 
                                    mean_params, std_params, std_max=None, std_min=None,
                                    shape_params=None, scale_params=None,
                                    reference_series=None):
        """Adds an extra point for a specific day in leap years.
        
        Args:
            points: Existing points array
            day: Day to add (should be 59 for Feb 29)
            decil_idx: Decil index for the day
            temp_type: 'mean', 'max', or 'min'
            mean_params, std_params, etc.: Temperature parameters
            reference_series: For tmax/tmin, the tmean series
            
        Returns:
            np.array: Updated points array
        """
        # Get reference value for consistency
        reference_value = None
        if reference_series is not None and day < len(reference_series):
            reference_value = reference_series[day]
        
        # Generate temperature for the extra day
        temperature = self._generate_temperature_for_day(
            day, decil_idx, temp_type, 
            mean_params, std_params, std_max, std_min,
            shape_params, scale_params, reference_value
        )
        
        # Create the new point
        new_point = [day, temperature]
        
        # Insert the point in the correct position (maintaining sorted order by day)
        points = np.vstack([points, new_point])
        points = points[points[:, 0].argsort()]
        
        return points
    
    def _ensure_leap_year_points(self, points, temp_type, mean_params, std_params, 
                                std_max, std_min, shape_params, scale_params,
                                reference_series=None):
        """Asegura que tenemos puntos adecuados para años bisiestos.
        
        Para años bisiestos, necesitamos asegurar que haya un punto cerca del 29 de febrero
        para una interpolación suave.
        """
        if not self.leap_year:
            return points
        
        # Día 59 es el 29 de febrero (0-based)
        feb_29_day = 59
        
        # Verificar si ya tenemos un punto cercano al 29 de febrero
        # Buscar el punto más cercano antes del 29 de febrero
        points_before = points[points[:, 0] < feb_29_day]
        
        if len(points_before) > 0:
            last_point_before = points_before[-1]
            day_before = int(last_point_before[0])
            
            # Si el último punto antes del 29 de febrero está demasiado lejos,
            # agregamos un punto en el 29 de febrero
            if feb_29_day - day_before > 7:  # Si hay más de 7 días de diferencia
                decil = self.get_decil(feb_29_day)
                idx = decil - 1
                
                points = self._add_february_leap_day_point(
                    points, feb_29_day, idx, temp_type,
                    mean_params, std_params, std_max, std_min,
                    shape_params, scale_params, reference_series
                )
        
        return points
    
    def generate_daily_temperature_with_noise(self, points, std_params, std_max, std_min,
                                              shape_params, scale_params, temp_type, 
                                              reference_series=None):
        """Generates a daily temperature series by linear interpolation with noise applied only to intermediate points.

        Args:
            points (np_array): Array of points [[days, temperature], ...]
            std_params: Standard deviation parameters for normal noise
            std_max: Standard deviation for maximum temperature
            std_min: Standard deviation for minimum temperature
            shape_params: Shape parameters for gamma distribution
            scale_params: Scale parameters for gamma distribution
            temp_type: 'mean', 'max', or 'min'
            reference_series: Series de referencia para verificar condiciones (tmean para tmax/tmin)
        
        Returns:
            Array with daily temperatures (total_days valores)
        """
        if len(points) < 2:
            raise ValueError("Must be at least two points")
        
        # Para años bisiestos, asegurar puntos adecuados
        if self.leap_year:
            points = self._ensure_leap_year_points(
                points, temp_type, None, std_params, std_max, std_min,
                shape_params, scale_params, reference_series
            )
        
        daily_temp = []
        
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i+1]
            x1, x2 = p1[0], p2[0]
            y1, y2 = p1[1], p2[1]
            
            m, b = self._linear_interpolation(p1, p2)
            
            n_points = int((x2 - x1))
            if n_points <= 0:
                continue
                
            xs = np.linspace(x1, x2, n_points, endpoint=False)
            ys = m * xs + b
            
            # Aplicar ruido solo a los puntos intermedios (no a los puntos originales)
            noisy_ys = []
            for j, (x, y) in enumerate(zip(xs, ys)):
                day = int(round(x))
                decil = self.get_decil(day)
                idx = decil - 1
                
                # Solo aplicar ruido si no es un punto original
                is_original_point = (abs(x - x1) < 1e-9 and j == 0) or (abs(x - x2) < 1e-9 and j == len(xs)-1)
                
                if not is_original_point:
                    if temp_type == 'mean':
                        # Ruido normal para temperatura media
                        noise = np.random.normal(0, std_params[idx])
                        temp_candidate = y + noise
                        # Aplicar límites a temperatura media
                        temp_candidate = self._apply_temperature_limits(temp_candidate)
                        noisy_ys.append(temp_candidate)
                    
                    elif temp_type == 'max':
                        # Ruido normal para temperatura máxima
                        for attempt in range(self.max_attempts):
                            noise = np.random.normal(0, std_max[idx])
                            temp_candidate = y + noise
                            
                            # Aplicar límite máximo
                            temp_candidate = self._apply_temperature_limits(temp_candidate, 'max')
                            
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
                            if attempt == self.max_attempts - 1:
                                if reference_series is not None and day < len(reference_series):
                                    enforced_temp = max(y, reference_series[day] + 0.5)
                                    enforced_temp = self._apply_temperature_limits(enforced_temp, 'max')
                                    noisy_ys.append(enforced_temp)
                                else:
                                    temp_candidate = self._apply_temperature_limits(temp_candidate, 'max')
                                    noisy_ys.append(temp_candidate)
                    
                    elif temp_type == 'min':
                        # Ruido normal para temperatura mínima
                        for attempt in range(self.max_attempts):
                            noise = np.random.normal(0, std_min[idx])
                            temp_candidate = y + noise
                            
                            # Aplicar límite mínimo
                            temp_candidate = self._apply_temperature_limits(temp_candidate, 'min')
                            
                            # Verificar que tmin < tmean
                            if reference_series is not None and day < len(reference_series):
                                if temp_candidate < reference_series[day]:
                                    noisy_ys.append(temp_candidate)
                                    break
                            else:
                                # Si no hay serie de referencia, aceptar cualquier valor
                                noisy_ys.append(temp_candidate)
                                break
                            
                            # Si llegamos al último intento, usar el valor aunque no cumpla
                            if attempt == self.max_attempts - 1:
                                if reference_series is not None and day < len(reference_series):
                                    enforced_temp = min(y, reference_series[day] - 0.5)
                                    enforced_temp = self._apply_temperature_limits(enforced_temp, 'min')
                                    noisy_ys.append(enforced_temp)
                                else:
                                    temp_candidate = self._apply_temperature_limits(temp_candidate, 'min')
                                    noisy_ys.append(temp_candidate)
                else:
                    # Mantener el punto original sin ruido, pero aplicar límites
                    original_temp = self._apply_temperature_limits(y, temp_type)
                    noisy_ys.append(original_temp)
            
            if noisy_ys:
                daily_temp.append(noisy_ys)
        
        # Agregar el último punto original (sin ruido, pero con límites aplicados)
        last_temp = self._apply_temperature_limits(points[-1][1], temp_type)
        daily_temp.append([last_temp])
        
        full_series = np.concatenate(daily_temp)
        
        # Asegurar que tenemos exactamente total_days días
        return full_series[:self.total_days]
    
    def _generate_temperature_point_with_limits(self, day, mean_params, std_params, 
                                               shape_params, scale_params):
        """Genera un punto de temperatura con validación de límites.
        
        Args:
            day (int): Día del año
            mean_params: Parámetros de media por decil
            std_params: Parámetros de desviación estándar por decil
            shape_params: Parámetros de forma gamma por decil
            scale_params: Parámetros de escala gamma por decil
            
        Returns:
            tuple: (mean_value, max_value, min_value) dentro de los límites
        """
        decil = self.get_decil(day)
        idx = decil - 1
        
        for attempt in range(self.max_attempts):
            # Generar valores de temperatura
            mean_value = np.random.normal(mean_params[idx], std_params[idx])
            variations = gamma.rvs(a=shape_params[idx], loc=0, scale=scale_params[idx])
            max_value = mean_value + variations
            min_value = mean_value - variations
            
            # Aplicar límites
            max_value = self._apply_temperature_limits(max_value, 'max')
            min_value = self._apply_temperature_limits(min_value, 'min')
            
            # Validar que max > min (consistencia física)
            if max_value > min_value:
                # Recalcular mean_value para mantener consistencia después de aplicar límites
                adjusted_mean = (max_value + min_value) / 2
                return adjusted_mean, max_value, min_value
        
        # Si no se encontraron valores válidos después de max_attempts
        # Usar valores por defecto dentro de los límites
        default_mean = (self.max_temp_limit + self.min_temp_limit) / 2
        default_max = self.max_temp_limit - 1
        default_min = self.min_temp_limit + 1
        
        return default_mean, default_max, default_min
    
    def generate_temperature_points(self, spacing, mean_params, std_params, std_max, std_min,
                                    shape_params, scale_params):
        """Generates temperature points for specific days according to the decile they belong to.
            
        Args:
            spacing (int): Espaciado entre días (2, 3, 5, etc.)
            mean_params: Mean temperature parameters per decil.
            std_params: Standard deviation parameters per decil.
            std_max: Standard deviation for maximum temperature per decil
            std_min: Standard deviation for minimum temperature per decil
            shape_params: Shape parameters for gamma distribution.
            scale_params: Scale parameters for gamma distribution.
            
        Returns:
            Arrays de puntos y series (total_days días)
        """
        # Generar días espaciados
        days_interval = self.generate_spaced_days(spacing)
        
        # Paso 1: Generar puntos iniciales para los días del intervalo CON LÍMITES
        p_tmean_initial = []
        p_tmax_initial = []
        p_tmin_initial = []
        
        for day in days_interval:
            mean_value, max_value, min_value = self._generate_temperature_point_with_limits(
                day, mean_params, std_params, shape_params, scale_params)
            
            p_tmean_initial.append([day, mean_value])
            p_tmax_initial.append([day, max_value])
            p_tmin_initial.append([day, min_value])
        
        # Convertir a arrays numpy
        p_tmean_initial = np.array(p_tmean_initial)
        p_tmax_initial = np.array(p_tmax_initial)
        p_tmin_initial = np.array(p_tmin_initial)
        
        # Paso 2: Primero generar serie de tmean (sin dependencias)
        t_tmean_final = self.generate_daily_temperature_with_noise(
            p_tmean_initial, std_params, std_max, std_min, 
            shape_params, scale_params, 'mean')
        
        # Para años bisiestos, asegurar puntos adecuados para tmax y tmin
        # NOTA: No agregamos puntos extra manualmente, dejamos que el método
        # _ensure_leap_year_points lo haga automáticamente si es necesario
        
        # Paso 3: Generar tmax y tmin usando tmean como referencia
        t_tmax_final = self.generate_daily_temperature_with_noise(
            p_tmax_initial, std_params, std_max, std_min,
            shape_params, scale_params, 'max', t_tmean_final)
        
        t_tmin_final = self.generate_daily_temperature_with_noise(
            p_tmin_initial, std_params, std_max, std_min,
            shape_params, scale_params, 'min', t_tmean_final)
        
        '''
        # Verificación de que las temperaturas en el 29 de febrero sean razonables
        if self.leap_year:
            feb_29_idx = 59
            if feb_29_idx < len(t_tmean_final):
                print(f"Temperatura 29 de febrero (día {feb_29_idx}):")
                print(f"  Tmean: {t_tmean_final[feb_29_idx]:.2f}°C")
                print(f"  Tmax: {t_tmax_final[feb_29_idx]:.2f}°C")
                print(f"  Tmin: {t_tmin_final[feb_29_idx]:.2f}°C")
                
                # Verificar consistencia
                if t_tmax_final[feb_29_idx] <= t_tmean_final[feb_29_idx]:
                    print(f"  ADVERTENCIA: Tmax <= Tmean en 29 de febrero")
                if t_tmin_final[feb_29_idx] >= t_tmean_final[feb_29_idx]:
                    print(f"  ADVERTENCIA: Tmin >= Tmean en 29 de febrero")
        '''
        # Verificación final de límites
        final_max = np.max(t_tmax_final)
        final_min = np.min(t_tmin_final)
        
        if final_max > self.max_temp_limit or final_min < self.min_temp_limit:
            print(f"Advertencia: Límites excedidos después de la interpolación")
            print(f"Máximo: {final_max:.2f}°C, Límite: {self.max_temp_limit}°C")
            print(f"Mínimo: {final_min:.2f}°C, Límite: {self.min_temp_limit}°C")
        
        return t_tmin_final, t_tmax_final, t_tmean_final
    
    def print_decil_distribution(self):
        """Muestra la distribución de días por decil para verificación"""
        print(f"Distribución de días por decil ({'Año bisiesto' if self.leap_year else 'Año no bisiesto'}):")
        print("Decil | Día inicio | Día fin | Cantidad días")
        print("-" * 45)
        
        for start, end, decil in self.decil_ranges:
            days_count = end - start + 1
            print(f"{decil:5} | {start:10} | {end:7} | {days_count:12}")
        
        # Verificar total de días
        total_days_calculated = sum([end - start + 1 for start, end, _ in self.decil_ranges])
        print(f"\nTotal de días cubiertos: {total_days_calculated}")
        print(f"Días esperados: {self.total_days}")
        print(f"Coincide: {total_days_calculated == self.total_days}")