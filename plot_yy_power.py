
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xx_power
import time

Tcmb = 2.728

def beam (ell, fwhm=0.5) :

    #convert fwhm from arcmin to radian
    fwhm *= (np.pi/180.0)/60.0
    sigma = fwhm / (np.sqrt(8.0*np.log(2.0)))
    bl = np.exp(ell*(ell+1.0) * sigma**2)

    return bl

def cl_to_clmuK2 ( power ) :

    return ( power * Tcmb**2.0 * 1.e6 )

def clmuK2_to_cl ( power ) :

    return ( power / (Tcmb**2.0 * 1.e6) )

def power (ell, theta, clump=True) :

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

    model = xx_power.return_yy_power(ell) # [erg cm^-2 s^-1 str^-1]^2

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
    opt = 1
    xx_power.init_cosmology(H0, Omega_M, Omega_b, w0, Omega_k, n_s, nH, inputPk, opt)

    shot_noise = 3.e-22

    ell = 10.**np.linspace(np.log10(10.),np.log10(3e4),40)

    theta_fid = [4.0, 3.e-5 ,0.026000,0.120000,1.000000,0.180000,0.800000,0.500000,0.100000,1.720000,0.195000,0.010000,0.800000,0.670000,0.730000,1.230000,0.880000, 3.85000]

    param_ind_dict = {'eps_f':0, 'eps_DM':1, 'f_star':2, 'S_star':3, 'A_C':4, 'alpha_nt':5, 'n_nt':6, 'beta_nt':7, 'gamma_mod0':8, 'gamma_mod_zslope':9, 'x_break':10, 'x_smooth':11, 'n_nt_mod':12, 'clump0':13, 'clump_zslope':14, 'x_clump':15, 'alpha_clump1':16, 'alpha_clump2':17}

    param_label_dict = {'eps_f':r'$\epsilon_f$', 'eps_DM':r'$\epsilon_{DM}$', 'f_star':r'$f_\star$', 'S_star':r'$S_\star$', 'A_C':r'$A_C$','alpha_nt':r'$\alpha_{nt}$', 'n_nt':r'$n_{nt}$', 'beta_nt':r'$\beta_{nt}$', 'gamma_mod0':r'$\Gamma_0$', 'gamma_mod_zslope':r'$\beta_\Gamma$', 'n_nt_mod':'$n_{nt,mod}$', 'clump0':r'$C_0$', 'clump_zslope':r'$\beta_C$','x_clump':r'$x_{C}$', 'alpha_clump1':r'$\alpha_{C1}$', 'alpha_clump2':r'$\alpha_{C2}$'}


    params = [ 'eps_f', 'f_star', 'S_star', 'alpha_nt', 'n_nt', 'beta_nt', 'gamma_mod0', 'gamma_mod_zslope', 'clump0', 'clump_zslope', 'x_clump', 'alpha_clump1', 'alpha_clump2' ]
    #param_list = ['eps_f', 'f_star', 'S_star', 'clump0', 'alpha0']
    #param_list = ['f_star', 'S_star', 'clump0']
    #param_list = ['gamma_mod0']
    params = ['eps_f']

    planck_ell, planck_cl, planck_var = read_data("../Planck/planck_mask_hfi_R2_small_ell.txt")
    planck_cl *= planck_ell*(planck_ell+1.)/(2.0*math.pi)
    planck_cl_err = np.sqrt(planck_var)*planck_ell*(planck_ell+1.)/(2.0*math.pi)


    for param in params :

        param_ind = param_ind_dict[param]
        param_fid = theta_fid[param_ind]
        print(param_fid)
        param_val_list = []
        color_list = ['C0', 'C1', 'C2', 'C3', 'C4']
       
        varlist = [0.1, 0.5, 1.0, 1.5, 2.0]
        #for i in [0.01, 0.1, 1.0, 1.67]:
        for i in [1.0]:
        #for i in varlist:
            param_val = param_fid * i
            #param_val = i
            param_val_list.append(param_val)


        f = plt.figure( figsize=(5,5) )
        ax = f.add_axes([0.18,0.16,0.75,0.75])

        #ax.errorbar(planck_ell, planck_cl, yerr = planck_cl_err, color='k', label=r"Planck")
        cl_list = []
        for counter ,param_val in enumerate(param_val_list) :
            theta = theta_fid.copy()
            theta[param_ind] = param_val

            start = time.time()
            cl = power (ell, theta)
            end = time.time()
            print("Elapsed time: %s" % (end - start))
            cl *= ell*(ell+1)/(2.0*math.pi)
            cl *= (1.e6 * Tcmb)**2
            #cl *= Tcmb**2.0 * 1.e6

            label_str = param_label_dict[param]+'$= %.3f $'% (param_val)
            if param == 'eps_f' :
                label_str = param_label_dict[param]+r'$= %.1f \times 10^{-6}$'% (param_val)
            ax.plot (ell, cl, ls = '-', color=color_list[counter], label = label_str)

        ax.set_xlim ( 10, 3e3 )
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$10^{12}\ell(\ell+1)C_{\ell}^{yy}/2\pi$')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='best')

        outname = param+'_yy_power.png'
        #outname = param+'_xx_power.png'
        f.savefig(outname)
        f.clf()


if __name__ == "__main__" :
    main()
