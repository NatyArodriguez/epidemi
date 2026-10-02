"""Variante experimental de TemperatureGeneratorWithNoiseStepLeap que corrige
el doble conteo del presupuesto de varianza entre el sorteo de anclas y el
ruido de los dias intermedios.

Ver notebooks/modelo_temp/notas_modelo_temp.txt (2026-10-01) para el analisis
completo: el generador original usa el mismo std_decada tanto para sortear
los puntos ancla (cada `spacing` dias) como para el ruido de cada dia
intermedio interpolado, lo que sobrestima la varianza diaria real en ~28-33%.

Esta variante NO toca el sorteo de anclas (se valido que spacing=4 con el
std completo reproduce bien la persistencia de largo plazo via DFA/Hurst).
Solo escala el ruido de los dias intermedios por

    escala = sqrt(2*f*(1-f)) * factor_correccion

donde f es la posicion fraccional del dia entre las dos anclas que lo
rodean. Se encontro ademas que spacing=3 (en vez del spacing=4 original)
da mejor ajuste de DFA y autocorrelacion una vez corregida la varianza
(barrido de spacing, notas_modelo_temp.txt 2026-10-01); factor_correccion=0.440
es el valor calibrado para spacing=3.

NO reemplaza a temp_generador.py (el generador "oficial" usado en
paper_temp/note9 y la grilla de deltas): es un modulo experimental,
pensado para la investigacion de la cadena de k sintetica (ver
notebooks/ajuste/notas_ajuste.txt).

Trade-off conocido (NO resuelto): corregir la varianza empeora la
autocorrelacion de corto plazo (lags 1-2) respecto del generador oficial
sin corregir, que por tener mas ruido decorrelaciona mejor pese a su
exceso de varianza. spacing=3 es el mejor punto encontrado hasta ahora,
no una solucion completa.
"""

import numpy as np

from epidemi.core.temp_generador import TemperatureGeneratorWithNoiseStepLeap

SPACING_RECOMENDADO = 3
FACTOR_CORRECCION_RECOMENDADO = 0.440


class TemperatureGeneratorWithNoiseStepLeap_VarFix(TemperatureGeneratorWithNoiseStepLeap):
    """TemperatureGeneratorWithNoiseStepLeap con el ruido de dias intermedios
    escalado para no duplicar el presupuesto de varianza del std_decada.

    Args:
        factor_correccion (float): factor multiplicativo adicional sobre
            sqrt(2*f*(1-f)), calibrado empiricamente para que la varianza
            diaria resultante coincida con la real. Default 0.440,
            calibrado para spacing=3 (ver notas_modelo_temp.txt).
    """

    def __init__(self, *args, factor_correccion=FACTOR_CORRECCION_RECOMENDADO, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor_correccion = factor_correccion

    def generate_daily_temperature_with_noise(self, points, std_params, std_max, std_min,
                                              shape_params, scale_params, temp_type,
                                              reference_series=None):
        if len(points) < 2:
            raise ValueError("Must be at least two points")

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

            noisy_ys = []
            for j, (x, y) in enumerate(zip(xs, ys)):
                day = int(round(x))
                decil = self.get_decil(day)
                idx = decil - 1

                is_original_point = (abs(x - x1) < 1e-9 and j == 0) or (abs(x - x2) < 1e-9 and j == len(xs)-1)

                if not is_original_point:
                    # reparto del presupuesto de varianza: cero pegado a un
                    # ancla, maximo en el punto medio del segmento
                    f = (x - x1) / (x2 - x1) if x2 != x1 else 0.0
                    escala = np.sqrt(2.0 * f * (1.0 - f)) * self.factor_correccion

                    if temp_type == 'mean':
                        noise = np.random.normal(0, std_params[idx] * escala)
                        temp_candidate = y + noise
                        temp_candidate = self._apply_temperature_limits(temp_candidate)
                        noisy_ys.append(temp_candidate)

                    elif temp_type == 'max':
                        for attempt in range(self.max_attempts):
                            noise = np.random.normal(0, std_max[idx] * escala)
                            temp_candidate = y + noise
                            temp_candidate = self._apply_temperature_limits(temp_candidate, 'max')
                            if reference_series is not None and day < len(reference_series):
                                if temp_candidate > reference_series[day]:
                                    noisy_ys.append(temp_candidate)
                                    break
                            else:
                                noisy_ys.append(temp_candidate)
                                break
                            if attempt == self.max_attempts - 1:
                                if reference_series is not None and day < len(reference_series):
                                    enforced_temp = max(y, reference_series[day] + 0.5)
                                    enforced_temp = self._apply_temperature_limits(enforced_temp, 'max')
                                    noisy_ys.append(enforced_temp)
                                else:
                                    temp_candidate = self._apply_temperature_limits(temp_candidate, 'max')
                                    noisy_ys.append(temp_candidate)

                    elif temp_type == 'min':
                        for attempt in range(self.max_attempts):
                            noise = np.random.normal(0, std_min[idx] * escala)
                            temp_candidate = y + noise
                            temp_candidate = self._apply_temperature_limits(temp_candidate, 'min')
                            if reference_series is not None and day < len(reference_series):
                                if temp_candidate < reference_series[day]:
                                    noisy_ys.append(temp_candidate)
                                    break
                            else:
                                noisy_ys.append(temp_candidate)
                                break
                            if attempt == self.max_attempts - 1:
                                if reference_series is not None and day < len(reference_series):
                                    enforced_temp = min(y, reference_series[day] - 0.5)
                                    enforced_temp = self._apply_temperature_limits(enforced_temp, 'min')
                                    noisy_ys.append(enforced_temp)
                                else:
                                    temp_candidate = self._apply_temperature_limits(temp_candidate, 'min')
                                    noisy_ys.append(temp_candidate)
                else:
                    original_temp = self._apply_temperature_limits(y, temp_type)
                    noisy_ys.append(original_temp)

            if noisy_ys:
                daily_temp.append(noisy_ys)

        last_temp = self._apply_temperature_limits(points[-1][1], temp_type)
        daily_temp.append([last_temp])

        full_series = np.concatenate(daily_temp)
        return full_series[:self.total_days]
