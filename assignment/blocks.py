import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def production_firm(par,ini,ss,Gamma,K,L,rK,w,Y):

    K_lag = lag(ini.K,K)

    # a. implied prices (remember K and L are inputs)
    rK[:] = par.alpha*Gamma*(K_lag/L)**(par.alpha-1.0)
    w[:] = (1.0-par.alpha)*Gamma*(K_lag/L)**par.alpha
    
    # b. production and investment
    Y[:] = Gamma*K_lag**(par.alpha)*L**(1-par.alpha)
    pass

@nb.njit
def mutual_fund(par,ini,ss,K,rK,A,r):

    # a. total assets
    A[:] = K

    # b. return
    r[:] = rK-par.delta
    pass

@nb.njit
def market_clearing(par,ini,ss,A,A_hh,L_low,L_high,L_hh_high,L_hh_low,Y,C_hh,K,I,clearing_A,clearing_L_low,clearing_L_high,clearing_Y):

    clearing_A[:] = A-A_hh
    clearing_L_low[:] = L_low-L_hh_low
    clearing_L_high[:] = L_high-L_hh_high
    I = K-(1-par.delta)*lag(ini.K,K)
    clearing_Y[:] = Y-C_hh-I
    pass
