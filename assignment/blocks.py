import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def production_firm(par,ini,ss,Gamma,K,rK,w_low,w_high,Y,phi_low,phi_high):

    K_lag = lag(ini.K,K)
    L_low = 2/3*phi_low
    L_high = 1/3*phi_high

    # a. implied prices 
    rK[:] = par.alpha*par.Gamma*K_lag**(par.alpha-1.0)*L_low**((1-par.alpha)/2)*L_high**((1-par.alpha)/2)
    w_low[:] = (1-par.alpha)/2*par.Gamma*K_lag**par.alpha*L_low**(-(par.alpha+1)/2)*L_high**((1-par.alpha)/2)
    w_high[:] = (1-par.alpha)/2*par.Gamma*K_lag**par.alpha*L_low**((1-par.alpha)/2)*L_high**(-(par.alpha+1)/2)

    
    # b. production and labor
    Y[:] = par.Gamma*K_lag**(par.alpha)*L_low**((1-par.alpha)/2)*L_high**((1-par.alpha)/2)


@nb.njit
def mutual_fund(par,ini,ss,K,rK,A,r):

    # a. total assets
    A[:] = K

    # b. return
    r[:] = rK-par.delta


@nb.njit
def market_clearing(par,ini,ss,A,A_hh,L_low,L_high,L_HIGH_hh,L_LOW_hh,Y,C_hh,K,I,clearing_A,clearing_L_low,clearing_L_high,clearing_Y):

    clearing_A[:] = A-A_hh
    clearing_L_low[:] = L_low-L_LOW_hh
    clearing_L_high[:] = L_high-L_HIGH_hh
    I = K-(1-par.delta)*lag(ini.K,K)
    clearing_Y[:] = Y-C_hh-I

