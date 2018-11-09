
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xx_power

flux_lim = 3.e-12

def power (ell) :
    '''
    double alpha0; // fiducial : 0.18
    double n_nt;   // fiducial : 0.80
    double beta;   // fiducial : 0.50
    double eps_f;  // fiducial : 3.97e-6
    double eps_DM; // fiducial : 0.00
    double f_star; // fiducial : 0.026
    double S_star; // fiducial : 0.12
    double A_C;    // fiducial : 1.00
    double gamma_mod0; // fiducial : 0.10
    double gamma_mod_zslope; // fiducial : 1.72
    double x_break; // fiducial : 0.195
    double x_smooth; // fiducial : 0.01
    double n_nt_mod; // fiducial : 0.80
    '''

    #Shaw model param
    eps_f = 3.97e-6
    eps_DM = 0.00
    f_star = 0.026
    S_star = 0.12
    A_C = 1.00
    gamma_mod0 = 0.10
    gamma_mod_zslope = 1.72

    #fix non-thermal pressure term
    alpha0 = 0.18
    n_nt = 0.80
    beta = 0.50
    x_smooth = 0.01
    n_nt_mod = 0.80
    x_break = 0.1

    #clumping terms
    clump0 = 0.0
    clump_zslope = 0.0
    x_clump = 1.0
    alpha_clump1 = 0.0
    alpha_clump2 = 0.0

    xx_power.set_Flender_params(alpha0, n_nt, beta, eps_f*1.e-6, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod, clump0, clump_zslope, x_clump, alpha_clump1, alpha_clump2)

    model = xx_power.return_xx_power(ell,flux_lim) # [erg cm^-2 s^-1 str^-1]^2
    #model = xx_power.return_xx_power(ell) # [erg cm^-2 s^-1 str^-1]^2

    return model

def main ():

    ell = 10.**np.linspace(1.,4.)

    cl = power (ell)

    print ell, cl

if __name__ == "__main__" :
    main()
