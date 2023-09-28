import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def production_firm(par,ini,ss,Gamma,K,L_low,L_high,rK,w_low,w_high,Y):

    K_lag = lag(ini.K,K)

    # a. implied prices 
    rK[:] = par.alpha*Gamma*K_lag**(par.alpha-1.0)*L_low**((1-par.alpha)/2)*L_high**((1-par.alpha)/2)
    w_low[:] = (1-par.alpha)/2*Gamma*K_lag**par.alpha*L_low**(-(par.alpha+1)/2)*L_high**((1-par.alpha)/2)
    w_high[:] = (1-par.alpha)/2*Gamma*K_lag**par.alpha*L_low**((1-par.alpha)/2)*L_high**(-(par.alpha+1)/2)

    
    # b. production and investment
    Y[:] = Gamma*K_lag**(par.alpha)*L_low**((1-par.alpha)/2)*L_high**((1-par.alpha)/2)

@nb.njit
def mutual_fund(par,ini,ss,K,rK,A,r):

    # a. total assets
    A[:] = K

    # b. return
    r[:] = rK-par.delta


@nb.njit
def market_clearing(par,ini,ss,A,A_hh,L_low,L_high,L_hh_high,L_hh_low,Y,C_hh,K,I,clearing_A,clearing_L_low,clearing_L_high,clearing_Y):

    clearing_A[:] = A-A_hh
    clearing_L_low[:] = L_low-L_hh_low
    clearing_L_high[:] = L_high-L_hh_high
    I = K-(1-par.delta)*lag(ini.K,K)
    clearing_Y[:] = Y-C_hh-I

