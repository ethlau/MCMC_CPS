
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xx_power
import time


def beam_transfer_function (ell, cl, fwhm) :

    #convert fwhm from arcmin to radian
    fwhm *= (np.pi/180.0)/60.0

    sigma = fwhm / (np.sqrt(8.0*np.log(2.0)))

    bl = np.exp(-ell**2 * sigma**2/2.0)

    return bl

def power (ell, theta, clump=False) :

    eps_f = theta[0]
    eps_DM = theta[1]
    f_star = theta[2]
    S_star = theta[3]
    A_C = theta[4]

    alpha0 = theta[5]
    n_nt = theta[6]
    beta = theta[7]

    gamma_mod0 = theta[8]
    gamma_mod_zslope = theta[9]
    x_break = theta[10]
    x_smooth = theta[11]
    n_nt_mod = theta[12]

    if clump :
        clump0 = theta[13]
        clump_zslope = theta[14]
        x_clump = theta[15]
        alpha_clump1 = theta[16]
        alpha_clump2 = theta[17]
    else :
        clump0 = 0.0
        clump_zslope = 0.0
        x_clump = 1.0
        alpha_clump1 = 0.0
        alpha_clump2 = 0.0


    xx_power.set_Flender_params(alpha0, n_nt, beta, eps_f*1e-6, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod, clump0, clump_zslope, x_clump, alpha_clump1, alpha_clump2)

    model = xx_power.return_xx_power(ell) # [erg cm^-2 s^-1 str^-1]^2

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
    nH = 2.4e+20

    xx_power.init_cosmology(H0, Omega_M, Omega_b, w0, Omega_k, n_s, nH, inputPk)

    shot_noise = 3.e-22

    ell = 10.**np.linspace(np.log10(1.),np.log10(3.e4), 1000)

    theta = [5.0,0.000000,0.026000,0.120000,1.000000,0.180000,0.800000,0.500000,0.100000,1.720000,0.195000,0.010000,0.800000,0.670000,0.730000,1.230000,0.880000,3.850000]

    param_ind_dict = {'eps_f':0, 'eps_DM':1, 'f_star':2, 'S_star':3, 'A_C':4, 'gamma_mod0':8, 'gamma_mod_zslope':9, 'clump0':13, 'clump_zslope':14}
    param_label_dict = {'eps_f':r'\epsilon_f', 'eps_DM':r'\epsilon_{DM}', 'f_star':r'f_\star', 'S_star':r'S_\star', 'A_C':r'A_C','gamma_mod0':r'\Gamma_0', 'gamma_mod_zslope':r'\beta_\Gamma', 'clump0':r'C_0', 'clump_zslope':r'\beta_C'}

    rosat_ell, rosat_cl, rosat_var = read_data("../ROSAT/rosat_R6_mask_hfi_R2_small_ell.txt")
    rosat_cl *= rosat_ell*(rosat_ell+1.)/(2.0*math.pi)
    rosat_cl_err = np.sqrt(rosat_var)*rosat_ell*(rosat_ell+1.)/(2.0*math.pi)

    theta_bf = [ 9.94086977e+00,  3.19781381e-02,  1.98917895e-01,  2.85712644e-03,
 -2.28851210e+01]

    theta[param_ind_dict['eps_f']] = theta_bf[0]
    theta[param_ind_dict['f_star']] = theta_bf[1]
    theta[param_ind_dict['S_star']] = theta_bf[2]
    theta[param_ind_dict['clump0']] = theta_bf[3]
    shot_noise = 10.0**theta_bf[4]

    f = plt.figure( figsize=(5,5) )
    ax = f.add_axes([0.18,0.16,0.75,0.75])

    cl = power (ell, theta)
    cl *= ell*(ell+1)/(2.0*math.pi)
    psn = np.full(ell.shape, shot_noise, dtype = np.float64)
    psn *=  ell*(ell+1)/(2.0*math.pi)
    total = cl + psn

    ax.plot (ell, total, ls = '-' )
    ax.plot (ell, cl, ls = '--')
    ax.plot (ell, psn, ls = ':')

    ax.errorbar(rosat_ell, rosat_cl, yerr = rosat_cl_err, color='k', label=r"ROSAT")

    ax.set_xlim ( 10, 3e4 )
    #ax.set_ylim ( 1e-19, 1e-13)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\ell(\ell+1)C_{\ell}/2\pi\,[{\rm erg^{2}s^{-2}cm^{-4}str^{-2}}]$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best')

    outname = '../plots/bf_xxpower.pdf'
    f.savefig(outname)
    f.clf()


if __name__ == "__main__" :
    main()
