"""Modelo eco-epidemiologico con las correcciones de la revision (oct 2026).

Misma estructura y mismas funciones que `utils.py` (constantes, funciones
auxiliares, integrador RK4 y `change_k`), salvo los siguientes cambios, que
responden a las objeciones del director:

A. Balance humano, con el criterio de `Main.c` (poblacion constante):
   - `dv[9]` (susceptibles) ya no resta `sigma_H*H_E`, que se resta solo en
     `dv[10]`. Antes la poblacion humana decrecia.
   - el 25% inmune (`1 - ALPHA`) arranca en `H_R`: `H_R0 = (1 - ALPHA)*poblacion`
     (antes `H_R0 = 0` y esa fraccion no estaba en ningun compartimento).
   - los importados entran a `H_I` y salen de `H_R` (`dv[12] = gama*H_I - deltaI`),
     como `Hr = H - Hs - He - Hi` en `Main.c`: H_S + H_E + H_I + H_R = poblacion.
B. Contador de casos: `G_T` aplica el mismo umbral que la ecuacion
   (`Tmin < NO_INFECCION`: sin infeccion vector -> humano) y usa `MIObh`
   (probabilidad de infeccion del humano) en lugar de `MIObv`.
D. Mortalidad acuatica por frio: con `Tmin < Temp_ACUATICA` sobrevive por dia
   la fraccion `SUPERVIVENCIA_ACUATICA` (0.6, `SUV` de `def_oran.c`) de huevos
   humedos, larvas y pupas: se agrega la perdida `(1 - SUV)*X` a la derivada.
   Antes se multiplicaba toda la derivada por `MUERTE_ACUATICA` (0.5), lo que
   frenaba la dinamica en lugar de aumentar la mortalidad.
E. Mortalidad de adultas por frio: con `Tmin < MATAR_VECTORES` (12.5 C) las
   adultas mueren a la tasa `MUERTE_FRIO_ADULTAS` (0.5/dia, `EFECT_V` de
   `def_oran.c`) en lugar de `2*mu_V`.

No se cambian (efecto chico en la prueba `ajuste/prueba_correcciones_modelo.ipynb`):
la formula de `KL`, el umbral de transmision (15 C) ni el umbral termico de
la infeccion del vector. Las condiciones iniciales son las de siempre, salvo
`H_R0` (ver A).

`utils.py` queda sin tocar: los resultados ya calibrados (k, deltas) se
hicieron con esa version.
"""
import numpy as np
import pandas as pd

from epidemi.core.utils import (
    load_data_file, load_data_file_pandas,
    MIObh, MIObv, bite_rate, EGG_LIFE, EGG_LIFE_wet, mu_Dry, mu_Wet, ALPHA,
    Remove_infect, Remove_expose, MATAR_VECTORES, NO_INFECCION, NO_LATENCIA,
    MUERTE_ACUATICA, Temp_ACUATICA, RATE_CASOS_IMP, MU_MOSQUITO_JOVEN,
    MADURACION_MOSQUITO, MU_MOSQUITA_ADULTA, Rthres, Hmax, kmmC, hogares,
    poblacion, oran,
    moving_average, suv_exp, Fbar, egg_wet, C_Gillet, hume, theta_T, rate_mx,
    rate_VE, muerte_V, calculo_EV, runge, cases,
)

# Fraccion de huevos humedos, larvas y pupas que sobrevive un dia con
# Tmin < Temp_ACUATICA (SUV de def_oran.c). Reemplaza a MUERTE_ACUATICA.
SUPERVIVENCIA_ACUATICA = 0.6

# Tasa de mortalidad de adultas en un dia con Tmin < MATAR_VECTORES (EFECT_V
# de def_oran.c). Reemplaza a mu_V = 2*mu_V.
MUERTE_FRIO_ADULTAS = 0.5


# ############################### MODELO PARA LA ODE ###############################
def modelo(v,t,EV,H_t,Tmean,Tmin,Rain,CasosImp,beta_day, Kmax):
    """Lado derecho de la ODE, con las correcciones A, D y E (ver docstring del modulo)."""

    dv = np.zeros(13)

    tt = int(t)

    E_D 	=	v[0]
    E_W		=	v[1]
    L		=	v[2]
    P		=	v[3]
    M		=	v[4]
    V		=	v[5]
    V_S		=	v[6]
    V_E		=	v[7]
    V_I		=	v[8]
    H_S		=	v[9]
    H_E		=	v[10]
    H_I		=	v[11]
    H_R		=	v[12]

    Tm      = Tmean[tt]

    rain    = Rain[tt]

    Tmin    = Tmin[tt]

    beta_day_theta_0 = beta_day*theta_T(Tm)

    fR      = egg_wet(rain)

    KL      = hogares*( Kmax*H_t/Hmax + 1.0 )

    m_E_C_G = 0.24*rate_mx(Tm, 10798.,100000.,14184.)*C_Gillet(L,KL)

    m_L     = 0.2088*rate_mx(Tm, 26018.,55990.,304.6)

    if (Tmin < 13.4):
        m_L = 0.

    mu_L    = 0.01 + 0.9725*np.exp(- (Tm - 4.85)/2.7035)

    C_L     = 1.5*(L/KL)

    m_P		=	0.384*rate_mx(Tm, 14931.,-472379.,148.)

    mu_P	=	0.01 + 0.9725*np.exp(- (Tm - 4.85)/2.7035)

    ### paramite modelo epi


    b_theta_pV	=	bite_rate*theta_T(Tm)*MIObv

    if ( Tm < NO_INFECCION):
        b_theta_pV = 0.

    mu_V    = muerte_V(Tm)*MU_MOSQUITA_ADULTA

    # E: mortalidad de adultas por frio
    if ( Tmin < MATAR_VECTORES):
        mu_V = MUERTE_FRIO_ADULTAS

    m_M     = MADURACION_MOSQUITO

    mu_M    = MU_MOSQUITO_JOVEN

    b_theta_pH		=	bite_rate*theta_T(Tm)*MIObh
    if ( Tmin < NO_INFECCION ):
        b_theta_pH = 0.

    sigma_H			=	1./Remove_expose
    gama			=	1./Remove_infect

    deltaI = RATE_CASOS_IMP*CasosImp[tt]

    # D: perdida diaria por frio en la fase acuatica (0 si Tmin >= Temp_ACUATICA)
    if (Tmin < Temp_ACUATICA):
        mu_frio = 1. - SUPERVIVENCIA_ACUATICA
    else:
        mu_frio = 0.

    ##modelo para la ODE

    dv[0]	=	beta_day_theta_0*V - fR*E_D - mu_Dry*E_D

    dv[1]	=	fR*E_D - m_E_C_G*E_W - mu_Wet*E_W - mu_frio*E_W

    dv[2]	=	m_E_C_G*E_W - m_L*L - ( mu_L + C_L )*L - mu_frio*L

    dv[3]	=	m_L*L - m_P*P - mu_P*P - mu_frio*P

    dv[4]	=	m_P*P - m_M*M - mu_M*M

    dv[5]	=	0.5*m_M*M - mu_V*V

    dv[6]	=	0.5*m_M*M - b_theta_pV*(H_I/poblacion)*V_S - mu_V*V_S

    dv[7]	=	b_theta_pV*(H_I/poblacion)*V_S - EV - mu_V*V_E

    dv[8]	=	EV - mu_V*V_I

    # A: los susceptibles solo pierden por infeccion
    dv[9]	=	- b_theta_pH*(H_S/poblacion)*V_I

    dv[10]	=	b_theta_pH*(H_S/poblacion)*V_I - sigma_H*H_E

    dv[11]	=	sigma_H*H_E - gama*H_I + deltaI

    # A: los importados salen de los recuperados (poblacion constante, como Main.c)
    dv[12]	=	gama*H_I - deltaI

    return dv


def nuevos_casos(Tmean_t, Tmin_t, v):
    """B: casos nuevos del dia (infecciones vector -> humano).

    Misma fuerza de infeccion que `dv[10]` en `modelo`: usa `MIObh` y es 0
    cuando `Tmin < NO_INFECCION`.
    """
    if Tmin_t < NO_INFECCION:
        return 0.
    return bite_rate*theta_T(Tmean_t) * MIObh * v[8] * v[9]/poblacion


def change_k(k,beta_day,temporada,i_date,initial_c=None,ci=None,tmin=None,
             rain=None, hr=None,
             tmean=oran[:,2], initial_H=None, devolver_H=False):
    """Simula la epidemia en la ventana `temporada` con k cambiando cada 1 de julio.

    Igual que `utils.change_k` (mismos argumentos y salidas), pero integra el
    `modelo` corregido de este modulo, cuenta los casos con `nuevos_casos`
    (correccion B) y, sin `initial_c`, arranca con el 25% inmune en `H_R`
    (correccion A). Si se pasa un `initial_c` de una corrida con `utils.py`,
    tendra `H_R` cerca de 0. Ademas, `tabla` se arma una sola vez al final y los
    importados por defecto se leen del paquete (`data/serie_ci_2001_2022.txt`,
    identico al de `notebooks/`), asi que funciona desde cualquier directorio.

    Args:
        k (list): capacidad de carga por temporada (avanza cada 30-jun -> 1-jul).
        beta_day (float): tasa de oviposicion.
        temporada (tuple): (fecha inicial, fecha final) de la simulacion.
        i_date (str): fecha del primer elemento de `tmean` (y de tmin/rain/hr
            si se pasan).
        initial_c (array, opcional): estado inicial (13 compartimentos).
        ci (tuple, opcional): (fecha, cantidad) de un unico caso importado; si
            es None se usa la serie de importados.
        tmin, rain, hr (array, opcional): si son None se toman de
            ORAN_2001_2022 indexados desde 2001-01-01.
        tmean (array): temperatura media, indexada desde `i_date`.
        initial_H (float, opcional): humedad del criadero inicial (None = 24).
        devolver_H (bool): si es True devuelve tambien la H_t final.

    Returns:
        (G_T, aedes, tabla) o, con devolver_H=True, (G_T, aedes, tabla, H_t final).
        Las salidas tienen un dia menos que la ventana: la fila t de `tabla` es
        el estado t dias despues de temporada[0] (la fila 0 es initial_c).
    """

    i_temporada = (np.datetime64(temporada[0]) - np.datetime64(i_date)).astype(int)
    f_temporada = (np.datetime64(temporada[1]) - np.datetime64(i_date)).astype(int) + 1

    i_ci = (np.datetime64(temporada[0]) - np.datetime64('2001-01-01')).astype(int)
    f_ci = (np.datetime64(temporada[1]) - np.datetime64('2001-01-01')).astype(int) + 1

    Tmean = tmean[i_temporada:f_temporada]

    if hr is None:
        HR = oran[i_ci:f_ci,4]
    else:
        HR = hr[i_temporada:f_temporada]

    if rain is None:
        Rain = oran[i_ci:f_ci,3]
    else:
        Rain = rain[i_temporada:f_temporada]

    if tmin is None:
        TMIN = oran[i_ci:f_ci,0]
    else:
        TMIN = tmin[i_temporada:f_temporada]

    DAYS = np.size(Tmean)

    if ci is None:
        casosIMP = load_data_file('serie_ci_2001_2022.txt')[i_ci:f_ci]
    else:
        casosIMP = cases(ci[0], ci[1], temporada)

    if initial_c is None:
        ED0  = 22876.
        EW0  = 102406
        L0   = 24962.
        P0   = 2003.
        M0   = 28836.
        V0   = 0.
        V_S0 = 0.
        V_E0 = 0.
        V_I0 = 0.
        H_S0 = ALPHA*poblacion
        H_E0 = 0.
        H_I0 = 0.
        H_R0 = poblacion - H_S0   # A: el 25% inmune va a H_R (Main.c)
        v = np.array([ED0, EW0, L0, P0, M0, V0, V_S0, V_E0, V_I0,
                      H_S0, H_E0, H_I0, H_R0], dtype=float)
    else:
        v = np.array(initial_c, dtype=float)[:13].copy()

    H_t  = 24. if initial_H is None else float(initial_H)

    dias = DAYS-1
    V_time = np.zeros((dias,13))
    V_time[0,:] = v

    aedes = np.empty(dias)
    aedes[0] = v[5]/poblacion

    G_T = np.empty(dias)
    G_T[0] = nuevos_casos(Tmean[0], TMIN[0], v)

    inicio = np.datetime64(temporada[0])
    fin = np.datetime64(temporada[1])
    fechas = np.arange(inicio, fin + 1, dtype='datetime64[D]')

    años = fechas.astype('datetime64[Y]')
    meses = fechas.astype('datetime64[M]')

    # Obtener mes y día numéricamente
    mes_num = (meses - años).astype(int) + 1
    dia_num = (fechas - meses).astype(int) + 1

    julio = np.where((mes_num == 6) & (dia_num == 30))[0]
    julio = julio[0:-1]

    valor = 0
    Kmax = k[valor]

    for t in range(1,dias):
        h = 1.
        G_T[t]	=  nuevos_casos(Tmean[t], TMIN[t], v)

        sigma_V     =	1./(1. + (0.1216*Tmean[int(t)]*Tmean[int(t)] - 8.66*Tmean[int(t)] + 154.79) )

        EV = sigma_V*v[7]
        H_t     = hume(H_t,Rain[int(t)], Tmean[int(t)], HR[int(t)])

        v = runge(modelo,t,h,v,args=(EV,H_t,Tmean,TMIN,Rain,casosIMP,beta_day,Kmax))

        v[v < 0.] = 0.

        V_time[t,:] = v
        aedes[t] = v[5]/poblacion

        if t in julio:
            valor = valor + 1
        Kmax = k[valor]

    labels = [
        "ED","EW","L","P","M","V",
        "V_S","V_E","V_I",
        "H_S","H_E","H_I","H_R"
    ]
    tabla = pd.DataFrame(V_time, columns=labels)

    if devolver_H:
        return G_T,aedes,tabla,H_t
    return G_T,aedes,tabla
