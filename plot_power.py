
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xx_power

flux_lim = 3.86e-13
#flux_lim = 0.0

def power (ell, clump=True, flux_lim=0.0) :
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
    eps_f = 1.0
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
    x_break = 0.195

    #0.180000 0.800000 0.500000 0.000001 0.000000 0.026000 0.120000 1.000000 0.100000 1.720000 0.195000 0.010000 0.800000 0.670000 0.730000 1.230000 0.880000 3.850000

    if clump :
        #clumping terms
        clump0 = 0.67
        clump_zslope = 0.73
        x_clump = 1.23
        alpha_clump1 = 0.88
        alpha_clump2 = 3.85
    else :
        clump0 = 0.0
        clump_zslope = 0.0
        x_clump = 0.0
        alpha_clump1 = 0.0
        alpha_clump2 = 0.0
 
    xx_power.set_Flender_params(alpha0, n_nt, beta, eps_f*1e-6, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod, clump0, clump_zslope, x_clump, alpha_clump1, alpha_clump2)

    model = xx_power.return_xx_power(ell,flux_lim) # [erg cm^-2 s^-1 str^-1]^2
    #model = xx_power.return_xx_power(ell) # [erg cm^-2 s^-1 str^-1]^2

    return model

def read_data (filename) :
    ell = []
    cl = []
    var = []
    with open(filename,'r') as f:
        f.readline()
        for line in f:
            cols = line.split(' ')
            ell.append(float(cols[0]))
            cl.append(float(cols[1]))
            var.append(float(cols[2]))

    ell = np.array(ell)
    cl = np.array(cl)
    var = np.array(var)

    return ell, cl, var

def main ():

    # set cosmology and linear power spectrum
    H0=70.0
    Omega_M=0.279000
    Omega_b=0.046100
    w0=-1.000000
    Omega_k=0.000000
    n_s=0.972000
    inputPk="../input_pk/wmap9_fid_matterpower_z0.dat"
    xx_power.init_cosmology(H0, Omega_M, Omega_b, w0, Omega_k, n_s, inputPk)

    ell = 10.**np.linspace(np.log10(10.),np.log10(3.e4), 100)
    print (ell)

    cl = power (ell, flux_lim=flux_lim)
 
    print(cl)
     
    cl *= ell*(ell+1)/(2.0*math.pi)

    cl_noclump = power (ell,clump=False,flux_lim = flux_lim)    
    cl_noclump *= ell*(ell+1)/(2.0*math.pi)

    plt.loglog(ell,cl,label='clump')
    plt.loglog(ell,cl_noclump, label=r'no clump')

    ell_obs,cl_obs,var_obs = read_data('../ROSAT/rosat_R4_R7_mask_hfi_R2_small_ell.txt')
    cl_obs *= ell_obs*(ell_obs+1)/(2.*math.pi)

    plt.loglog(ell_obs, cl_obs, label=r'ROSAT')

    plt.legend()
    plt.show()
    
    
if __name__ == "__main__" :
    main()
