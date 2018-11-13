
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xx_power
import time

flux_lim = 3.86e-13

def power (ell, theta, clump=True, flux_lim=0.0) :

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

    clump0 = theta[13]
    clump_zslope = theta[14]
    x_clump = theta[15]
    alpha_clump1 = theta[16]
    alpha_clump2 = theta[17]

    xx_power.set_Flender_params(alpha0, n_nt, beta, eps_f*1e-6, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod, clump0, clump_zslope, x_clump, alpha_clump1, alpha_clump2)

    model = xx_power.return_xx_power(ell,flux_lim) # [erg cm^-2 s^-1 str^-1]^2

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

    theta_fid = [1.0,0.000000,0.026000,0.120000,1.000000,0.180000,0.800000,0.500000,0.100000,1.720000,0.195000,0.010000,0.800000,0.670000,0.730000,1.230000,0.880000,3.850000]
   
    param_list = ['eps_f', 'gamma_mod0', 'gamma_mod_zslope', 'clump0', 'clump_zslope']
    param_ind_dict = {'eps_f':0, 'gamma_mod0':8, 'gamma_mod_zslope':9, 'clump0':13, 'clump_zslope':14}
    param_label_dict = {'eps_f':r'\eps_f', 'gamma_mod0':r'\Gamma_0', 'gamma_mod_zslope':r'\beta_\Gamma', 'clump0':r'C_0', 'clump_zslope':r'\beta_C'}
    
    for param in param_ind_dict.keys() :

        param_ind = param_ind_dict[param]        
        param_fid = theta_fid[param_ind]
        print(param_fid)
        param_list = []
        for i in [0.1, 0.5, 1.0, 1.5, 2.0]:
            param_val = param_fid * i
            param_list.append(param_val)

        print(param_list)        

        f = plt.figure( figsize=(5,5) )
        ax = f.add_axes([0.18,0.16,0.75,0.75])

        cl_list = []
        for param_val in param_list :
            theta = theta_fid
            theta[param_ind] = param_val

            start = time.time() 
            cl = power (ell, theta, flux_lim=flux_lim)
            end = time.time()
            print("Elapsed time: %s" % (end - start))
            cl *= ell*(ell+1)/(2.0*math.pi)
            cl_list.append(cl)

            label_str = r'$'+param_label_dict[param]+'= %.2e $'% (param_val) 
            ax.plot (ell, cl, ls = '-', label = label_str)

        ax.set_xlim ( 10, 3e4 )
        ax.set_ylim ( 1e-19, 1e-13)
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$\ell(\ell+1)C_{\ell,XX}/2\pi\,[{\rm erg^{2}s^{-2}cm^{-4}str^{-2}}]$')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='best')

        outname = param+'_power.pdf'
        f.savefig(outname)
        f.clf()
   
    
if __name__ == "__main__" :
    main()
