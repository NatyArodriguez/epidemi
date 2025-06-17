import numpy as np
import scipy.special as sc

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
            count_s         = 	j #importante
            sigma_U_T[j]    = 	sigma_V_1*Fbar(t, count_s , k_VE_1, theta_VE_1)*suv_exp(t, count_s, mu_V_1)
            mu_U_T[j]       = 	mu_V_1*Fbar(t, count_s , k_VE, theta_VE)*suv_exp(t, count_s, mu_V_1)
            U_T[j]          = 	Fbar(t, count_s , k_VE, theta_VE)*suv_exp(t, count_s, mu_V_1)
            
        for j in range(1,tt):
            #print(j)
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
