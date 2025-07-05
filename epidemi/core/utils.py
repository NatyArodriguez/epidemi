import numpy as np
import scipy.special as sc
import importlib.resources as pkg_resources

def load_data_file(filename):
    with pkg_resources.files('epidemi.data').joinpath(filename).open('r') as f:
        return np.loadtxt(f, skiprows=1)

MIObh         = 0.75
MIObv         = 0.75
bite_rate = 0.27
# vida de los huevos [dias]
EGG_LIFE            = 120.
EGG_LIFE_wet        = 90.
mu_Dry              = 1./EGG_LIFE
mu_Wet              = 1./EGG_LIFE_wet
ALPHA               = 0.75
Remove_infect       = 7.#//7. //dias 10
Remove_expose       = 5.#//7. //dias 829
MATAR_VECTORES      = 12.5#//9.5//12.5//temp
NO_INFECCION        = 15. #//15. //temp
NO_LATENCIA         = 16. #
MUERTE_ACUATICA     = 0.5
Temp_ACUATICA       = 10.
RATE_CASOS_IMP      = 1. #entero
# paramites aedes en days 
MU_MOSQUITO_JOVEN   = 1./2. 
MADURACION_MOSQUITO = 1./2.          
MU_MOSQUITA_ADULTA  = 1./10. #// vida^{-1} del vector en optimas condiciones (22 grados) 0.091; //
Rthres              = 12.5 
Hmax                = 24.
kmmC                = 3.3E-6
H_t  = 24.
hogares = 17633
poblacion = 75697

# ###FUNCIONES A UTILIZAR ####FUNCIONES A UTILIZAR ####FUNCIONES A UTILIZAR

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def suv_exp(t,s,rate):
    
    salida = 0.
    tau    = t - s
    
    if ( t > 0. ):
        salida = np.exp(-rate*tau)
    
    return salida

def Fbar(t,s,k,theta):
    salida = 0.
    tau = t - s
    
    if (t > 0. ):
        
        salida = sc.gammaincc(k, tau/theta)
        
    return salida
    
# fR fraccion de ED -> EW
def egg_wet(rain):
    salida = 0.
    lluvia   = rain / Rthres 
    salida = 0.8*lluvia**5/(1. + lluvia**5)
    
    return salida

# modelo Gillet
def C_Gillet(Larva,KL):
    salida = 0.
    if (Larva < KL):
        salida = 1. - Larva/KL
    else:
        salida = 0.
    
    return salida

# H(t+1)
def hume(H_t,rain,Tm,Hum):
    
    salida = H_t + rain - kmmC*(100. - Hum)*(25. + Tm)*(25. + Tm)
    
    if ( salida < 0.):
        salida = 0.
    
    if (Hmax < salida): 
        salida = Hmax
    
    return salida

# actividad
def theta_T(Tm):
    salida = 0.
    
    if ( (11.7 < Tm) and (Tm < 32.7)) :
        salida = 0.1137*(-5.4 + 1.8*Tm - 0.2124*Tm*Tm + 0.01015*Tm*Tm*Tm - 0.0001515*Tm*Tm*Tm*Tm);
    
    return salida

# tasas de maduracion L,P,M
def rate_mx(Tm,DHA,DHH,T12):
    salida = 0.
    aux0 = (DHA/1.987207) * (1./298. - 1./(Tm + 273.15) )
    aux1 = (DHH/1.987207) * (1./T12 - 1./(Tm + 273.15) )
    salida = ( (Tm + 273.15)/298. ) * ( np.exp( aux0 )  / (1.+ np.exp(aux1)) )

    return salida

# tasa de latencia mosquitos
def rate_VE(Tm,k):
    salida = k/np.exp( -0.155*Tm + 6.5031 )
    if (k<=3):
        salida = k/np.exp( -0.1659*Tm + 6.7031)
    
    return salida

# muerte de vectores
def muerte_V(Tm):
    salida = 0. 
    factor = 0.0360827 # factor a 22 grados
    salida = 8.692E-1 -(1.59E-1)*Tm + (1.116E-2)*Tm*Tm -(3.408E-4)*Tm*Tm*Tm + (3.809E-6)*Tm*Tm*Tm*Tm;
    salida = salida/factor
    
    return salida

def calculo_EV(t,Tm,V_E,G_T):
    salida = 0.
    tt = int(t)
    ##Calculo de las integrales
    #/* calculo EV */
    
    media_VE    = 1. + (0.1216*Tm*Tm - 8.66*Tm + 154.79) 
    var_VE      = 1. + (0.1728*Tm*Tm - 12.36*Tm + 230.62) 
    sigma_V     =	1./media_VE 
    k_VE        =	( media_VE*media_VE)/var_VE 
    theta_VE    =	var_VE / media_VE
    mu_V		=	muerte_V(Tm)*MU_MOSQUITA_ADULTA
    
    integral_1 = 0.
    integral_2 = 0.
    integral_3 = 0.
    integral_4 = 0.
    
    sigma_U_T  = np.zeros(tt)
    sigma_V_1  = np.zeros(tt) 
    mu_U_T     = np.zeros(tt)
    U_T        = np.zeros(tt)
    

    
    if (t > 2.):
        for j in range(0,tt):
            T_1             = Tm[j]
            media_VE_1      =	0.1216*T_1*T_1 - 8.66*T_1 + 154.79
            var_VE_1        =	0.1728*T_1*T_1 - 12.36*T_1 + 230.62
            sigma_V_1       =	1./media_VE_1
            k_VE_1          =	( media_VE_1*media_VE_1)/var_VE_1
            theta_VE_1      =	var_VE_1 / media_VE_1
            mu_V_1          =	muerte_V(T_1)*MU_MOSQUITA_ADULTA
            count_s         = 	j # importante
            sigma_U_T[j]    = 	sigma_V_1*Fbar(t, count_s , k_VE_1, theta_VE_1)*suv_exp(t, count_s, mu_V_1)
            mu_U_T[j]       = 	mu_V_1*Fbar(t, count_s , k_VE, theta_VE)*suv_exp(t, count_s, mu_V_1)
            U_T[j]          = 	Fbar(t, count_s , k_VE, theta_VE)*suv_exp(t, count_s, mu_V_1)
            
        for j in range(1,tt):
            # print(j)
            T_1	            =	Tm[int(j-1)] 
            media_VE_1		=	0.1216*T_1*T_1 - 8.66*T_1 + 154.79
            sigma_V_1		=	1./media_VE_1 
            mu_V_1			=	muerte_V(T_1)*MU_MOSQUITA_ADULTA
            T_2				=	Tm[int(j)]
            media_VE_2		=	0.1216*T_2*T_2 - 8.66*T_2 + 154.79 
            sigma_V_2		=	1./media_VE_2 
            mu_V_2			=	muerte_V(T_2)*MU_MOSQUITA_ADULTA
            
            integral_1		=	integral_1 + 0.5*( G_T[j-1]*U_T[j-1] + G_T[j]*U_T[j] )
            integral_2		=	integral_2 + 0.5*( sigma_V_1*G_T[j-1]*U_T[j-1] + sigma_V_2*G_T[j]*U_T[j] )
            integral_3		=	integral_3 + 0.5*( G_T[j-1]*U_T[j-1] + G_T[j]*U_T[j] )
            integral_4		=	integral_4 + 0.5*( mu_V_1*G_T[j-1]*U_T[j-1] + mu_V_2*G_T[j]*U_T[j] )
    
    
    if ( integral_1 < 0.): 
        integral_1 = 0.
    if ( integral_2 < 0.):
        integral_2 = 0.
    if ( integral_3 < 0.):
        integral_3 = 0.
    if ( integral_4 < 0.):
        integral_4 = 0.
         
              
    salida 	=	sigma_V*V_E - sigma_V*integral_1 + integral_2 - mu_V*integral_3 + integral_4
    
    if ( salida < 0. ): 
        salida = 0
        
    if(Tm < NO_LATENCIA):
        salida = 0. 
    
    return salida

def runge(f,t,h,X,args=()):
    n = len(X)
    salida = np.zeros(n)
    
    k1 = np.zeros(n)
    k2 = np.zeros(n)
    k3 = np.zeros(n)
    k4 = np.zeros(n)
    
    k1 = h*f(X, t,*args)
    k2 = h*f(X + 0.5*k1, t + 0.5*h,*args)
    k3 = h*f(X + 0.5*k2, t + 0.5*h,*args)
    k4 = h*f(X + k3, t + h,*args)

    salida = X + (1./6.)*( k1 + 2.*k2 + 2.*k3 + k4 )
    
    return salida


def cases(date, amount, season):
    u = np.datetime64(season[0])
    v = np.datetime64(season[1])
    z = (v-u).astype(int) + 1
    
    enter = (np.datetime64(date)-u).astype(int)
    
    aux = np.zeros(z)
    aux[enter] = amount
    return aux



# ############################### MODELO PARA LA ODE ###############################
def modelo(v,t,EV,H_t,Tmean,Tmin,Rain,CasosImp,beta_day, Kmax):
    
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
    
    if ( Tmin < MATAR_VECTORES):
        mu_V = 2*mu_V
        
    m_M     = MADURACION_MOSQUITO
    
    mu_M    = MU_MOSQUITO_JOVEN
		
    b_theta_pH		=	bite_rate*theta_T(Tm)*MIObh 
    if ( Tmin < NO_INFECCION ): 
        b_theta_pH = 0.
        
    sigma_H			=	1./Remove_expose 
    gama			=	1./Remove_infect
    
    #deltaI = RATE_CASOS_IMP*casosImp[week]
    deltaI = RATE_CASOS_IMP*CasosImp[tt]
    
    ##modelo para la ODE
    
    dv[0]	=	beta_day_theta_0*V - fR*E_D - mu_Dry*E_D
    
    dv[1]	=	fR*E_D - m_E_C_G*E_W - mu_Wet*E_W
        
    if (Tmin < Temp_ACUATICA):
        dv[1] = MUERTE_ACUATICA*dv[1]
    
    dv[2]	=	m_E_C_G*E_W - m_L*L - ( mu_L + C_L )*L
        
    if (Tmin < Temp_ACUATICA):
        dv[1] = MUERTE_ACUATICA*dv[2]
        
    dv[3]	=	m_L*L - m_P*P - mu_P*P
        
    if (Tmin < Temp_ACUATICA):
        dv[1] = MUERTE_ACUATICA*dv[3]
    
    dv[4]	=	m_P*P - m_M*M - mu_M*M
    
    dv[5]	=	0.5*m_M*M - mu_V*V
    
    dv[6]	=	0.5*m_M*M - b_theta_pV*(H_I/poblacion)*V_S - mu_V*V_S
    
    dv[7]	=	b_theta_pV*(H_I/poblacion)*V_S - EV - mu_V*V_E
    
    dv[8]	=	EV - mu_V*V_I
    
    dv[9]	=	- b_theta_pH*(H_S/poblacion)*V_I - sigma_H*H_E
    
    dv[10]	=	b_theta_pH*(H_S/poblacion)*V_I - sigma_H*H_E
    
    dv[11]	=	sigma_H*H_E - gama*H_I + deltaI
    
    dv[12]	=	gama*H_I
    
    return dv

days = np.arange('2001-01-01','2023-01-01', dtype='datetime64[D]')
oran = load_data_file('ORAN_2001_2022.txt')

def fun(k,beta,temporada,suma,ci=None,rain=oran[:,3],tmin=oran[:,0],tmean=oran[:,2],hr=oran[:,4]):
    
    i_temporada = (np.datetime64(temporada[0]) - days[0]).astype(int)
    f_temporada = (np.datetime64(temporada[1]) - days[0]).astype(int) + 1
    
    i_suma = (np.datetime64(suma[0]) - days[0]).astype(int)
    f_suma = (np.datetime64(suma[1]) - days[0]).astype(int) + 1

    TMIN = tmin[i_temporada:f_temporada]
    Tmean = tmean[i_temporada:f_temporada]
    Rain = rain[i_temporada:f_temporada]
    HR = hr[i_temporada:f_temporada]

    DAYS=np.size(Tmean)
    WEEKS = int(len(Tmean)/7) + 1
    
    if ci is None:
        casosIMP = load_data_file('serie_ci_2001_2022.txt')[i_temporada:f_temporada]
    else: 
        ingreso_ci = ci[0]
        cantidad_ci = ci[1]
        casosIMP = cases(ingreso_ci,cantidad_ci,temporada)
    
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
    H_R0 = ALPHA*poblacion - H_S0
    H_t  = 24.

    v = np.zeros(13)
    v[0]  = ED0
    v[1]  = EW0
    v[2]  = L0
    v[3]  = P0
    v[4]  = M0
    v[5]  = V0
    v[6]  = V_S0
    v[7]  = V_E0
    v[8]  = V_I0
    v[9]  = H_S0
    v[10] = H_E0
    v[11] = H_I0
    v[12] = H_R0

    dias = DAYS-1
    paso_d = np.zeros(dias)
    paso_w = np.zeros(WEEKS)
    solucion = np.empty_like(v)

    V_H = np.empty_like(paso_d)
    V_H_w = np.zeros_like(paso_w)
    egg_d = np.empty_like(paso_d)
    egg_w = np.empty_like(paso_d)
    larv  = np.empty_like(paso_d)
    pupa  = np.empty_like(paso_d)
    mosco = np.empty_like(paso_d)
    aedes = np.empty_like(paso_d)
    vec_s = np.empty_like(paso_d)
    vec_i = np.empty_like(paso_d)
    host_i = np.empty_like(paso_d)
    parametro   = np.zeros_like(paso_d)
    parametro_w = np.zeros_like(paso_w)

    egg_d[0] = v[0]/poblacion#egg_wet(Rain[0])#v[0]/poblacion #mu_Dry*
    egg_w[0] = v[1]/poblacion
    larv[0]  = v[2]/poblacion
    pupa[0]  = v[3]/poblacion
    mosco[0] = v[4]/poblacion
    aedes[0] = v[4]/poblacion
    vec_s[0] = v[4]/poblacion
    vec_i[0] = v[8]/poblacion
    host_i[0]= 0.
    parametro[0] = 0.

    G_T = np.empty_like(paso_d)
    G_TV = np.empty_like(paso_d)
    F_T = np.empty_like(paso_d)
    G_T[0] = bite_rate*theta_T(Tmean[0]) * MIObv * v[8] * v[9]/poblacion
    G_TV[0] = bite_rate*theta_T(Tmean[0]) * MIObv * v[6] * v[11]/poblacion
    F_T[0] = bite_rate*theta_T(Tmean[0]) * MIObh * v[8] * v[6]/poblacion

    G_T_week = np.zeros(WEEKS)
    G_TV_week = np.zeros(WEEKS)
    contar = 0 
    week = 0

    for t in range(1,dias):
        paso_d[int(t)] = t
        h = 1.

        G_T[t]	=  bite_rate*theta_T(Tmean[t])* MIObv * v[8] * v[9]/poblacion
        G_TV[t]	=  bite_rate*theta_T(Tmean[t])* MIObv * v[6] * v[11]/poblacion
        F_T[t] = bite_rate*theta_T(Tmean[0]) * MIObh * v[8] * v[6]/poblacion

        sigma_V     =	1./(1. + (0.1216*Tmean[int(t)]*Tmean[int(t)] - 8.66*Tmean[int(t)] + 154.79) )

        EV = sigma_V*v[7]
        H_t     = hume(H_t,Rain[int(t)], Tmean[int(t)], HR[int(t)])
        
        solucion     =  runge(modelo,t,h,v,args=(EV,H_t,Tmean,TMIN,Rain,casosIMP,beta,k))
        v = solucion

        for q in range(13):
            if (v[q]<0.):
                v[q] = 0.

        V_H[t] = v[0]/poblacion

        egg_d[t] = v[0]/poblacion
        egg_w[t] = v[1]/poblacion
        larv[t] = v[2]/poblacion
        pupa[t] = v[3]/poblacion
        mosco[t] = v[4]/poblacion
        aedes[t] = v[5]/poblacion
        vec_s[t] = v[6]/poblacion
        vec_i[t] = v[8]/poblacion
        
        gamma = 1./Remove_infect

        parametro[int(t)] =  np.sqrt( (sigma_V/gamma) * ((bite_rate*bite_rate*theta_T(Tmean[t])*theta_T(Tmean[t])*MIObh*MIObv)/(muerte_V(Tmean[t])*MU_MOSQUITA_ADULTA*(sigma_V + muerte_V(Tmean[t])*MU_MOSQUITA_ADULTA))) * vec_s[t] * ALPHA )

        G_T_week[week] = G_T_week[week] + G_T[t]
        #G_TV_week[week] = G_TV_week[week] + G_TV_week[t]

        contar = contar + 1
        if (contar > 6):
            G_T_week[week] = G_T_week[week]
            #G_TV_week[week] = G_TV_week[week]
            week = week + 1
            contar = 0
        
        host_i[t] = v[11]
        
    salida = np.sum(G_T[i_suma-1:f_suma])
    
    return G_T, salida

