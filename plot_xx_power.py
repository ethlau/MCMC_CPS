
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xx_power
import time
import healpy as hp

keV2erg = 1.6022e-9
deg2rad = math.pi/180.0
ster2sqdeg = 3282.80635
ster2sqarcmin = ster2sqdeg * 3600.0
ster2sqarcsec = ster2sqdeg * 3600.0 * 3600.0


def read_kolodzig (filename = "../Kolodzig/PS_DATA_LSS_RS09_0.5_2.0_NCR03.txt") :

    k, ell, psf, cl, cl_err = np.loadtxt (filename, usecols=(0,1,2,6,7), unpack = True)

    cl *= ster2sqdeg
    cl_err *= ster2sqdeg

    return  k, ell, psf, cl, cl_err


def beam (ell, fwhm=12.0) :

    #convert fwhm from arcmin to radian
    fwhm = math.radians(fwhm/60.0)
    sigma = fwhm / (np.sqrt(8.0*np.log(2.0)))
    bl = np.exp(ell*(ell+1.0) * sigma**2)

    return bl

def power (ell, theta, clump=True) :

    eps_f = theta[0]
    eps_DM = theta[1]
    f_star = theta[2]
    S_star = theta[3]
    A_C = theta[4]

    A_nt = theta[5]
    B_nt = theta[6]
    gamma_nt = theta[7]

    gamma_mod0 = theta[8]
    gamma_mod_zslope = theta[9]
    x_break = theta[10]
    x_smooth = theta[11]
    n_nt_mod = theta[12]

    clump0 = theta[13]
    alpha_clump = theta[14]
    beta_clump = theta[15]
    gamma_clump = theta[16]

    xx_power.set_Flender_params(eps_f*1e-6, eps_DM, f_star, S_star, A_C, A_nt, B_nt, gamma_nt, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod, clump0, alpha_clump, beta_clump, gamma_clump)

    #model = xx_power.return_xx_power(ell) # [erg cm^-2 s^-1 str^-1]^2
    model_alt= xx_power.return_xx_power_alt(ell) # [erg cm^-2 s^-1 str^-1]^2

    return model_alt

def cxb (theta) :

    eps_f = theta[0]
    eps_DM = theta[1]
    f_star = theta[2]
    S_star = theta[3]
    A_C = theta[4]

    A_nt = theta[5]
    B_nt = theta[6]
    gamma_nt = theta[7]

    gamma_mod0 = theta[8]
    gamma_mod_zslope = theta[9]
    x_break = theta[10]
    x_smooth = theta[11]
    n_nt_mod = theta[12]

    clump0 = theta[13]
    alpha_clump = theta[14]
    beta_clump = theta[15]
    gamma_clump = theta[16]

    xx_power.set_Flender_params(eps_f*1e-6, eps_DM, f_star, S_star, A_C, A_nt, B_nt, gamma_nt, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod, clump0, alpha_clump, beta_clump, gamma_clump)

    return (xx_power.return_total_xsb()/(4.0*math.pi))


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
    '''
    H0=70.0
    Omega_M=0.279000
    Omega_b=0.046100
    w0=-1.000000
    Omega_k=0.000000
    n_s=0.972000
    inputPk="../input_pk/wmap9_fid_matterpower_z0.dat"
    nH = 2.4e21
    opt = 1
    '''
    H0=67.32117
    Omega_M=0.3158
    Omega_b=0.0490
    w0=-1.000000
    Omega_k=0.000000
    n_s=0.96605
    inputPk="../input_pk/planck_2018_test_matterpower.dat"
    nH = 0.0
    opt = 0

    xx_power.init_cosmology(H0, Omega_M, Omega_b, w0, Omega_k, n_s, nH, inputPk, opt)

    ell = 10.**np.linspace(np.log10(10.),np.log10(3.e3),31)

    theta_fid = [4.0, 3.e-5 ,0.0250,0.120000,1.000000,0.180000,0.800000,0.500000,0.10000,1.720000,0.195000,0.010000,0.800000, 0.2, 1.0, 6.0, 3.0]

    param_ind_dict = {'eps_f':0, 'eps_DM':1, 'f_star':2, 'S_star':3, 'A_C':4, 'alpha_nt':5, 'n_nt':6, 'beta_nt':7, 'gamma_mod0':8, 'gamma_mod_zslope':9, 'x_break':10, 'x_smooth':11, 'n_nt_mod':12, 'clump0':13, 'alpha_clump':14, 'beta_clump':15, 'gamma_clump':16}

    param_label_dict = {'eps_f':r'$\epsilon_f$', 'eps_DM':r'$\epsilon_{DM}$', 'f_star':r'$f_\star$', 'S_star':r'$S_\star$', 'A_C':r'$A_C$','alpha_nt':r'$\alpha_{nt}$', 'n_nt':r'$n_{nt}$', 'beta_nt':r'$\beta_{nt}$', 'gamma_mod0':r'$\Gamma_0$', 'gamma_mod_zslope':r'$\beta_\Gamma$', 'n_nt_mod':'$n_{nt,mod}$', 'clump0':r'$C_0$', 'alpha_clump':r'$\alpha_C$','beta_clump':r'$\beta_{C}$', 'gamma_clump':r'$\gamma_{C}$'}

    rosat_ell, rosat_cl, rosat_var = read_data("../ROSAT/rosat_R4_R7_counts.txt")
    rosat_cl *= rosat_ell*(rosat_ell+1.)/(2.0*math.pi)
    rosat_cl_err = np.sqrt(rosat_var)
    rosat_cl_err *= rosat_ell*(rosat_ell+1.)/(2.0*math.pi)
    k_k, ell_k, beam_k, cl_k, cl_k_err = read_kolodzig ()
    cl_k /= beam_k
    cl_k *= ell_k*(ell_k+1)/(2.0*math.pi)
 
    #params = [ 'eps_f', 'f_star', 'S_star', 'alpha_nt', 'n_nt', 'beta_nt', 'gamma_mod0', 'gamma_mod_zslope', 'clump0', 'alpha_clump', 'beta_clump', 'gamma_clump' ]
    #params = [ 'eps_f', 'f_star', 'S_star', 'clump0', 'alpha_clump', 'beta_clump', 'gamma_clump' ]
    params = [ 'eps_f' ]
    param_values = {
        'eps_f':[4.0 ], 
        #'eps_f':[1.0, 2.0, 4.0, 6.0, 8.0 ], 
        'f_star':[0.01, 0.015, 0.02, 0.025, 0.03],
        'S_star':[0.03, 0.06, 0.12, 0.24, 0.48],
        'clump0':[0.01, 0.1, 1.0, 10.0, 100.0],
        'alpha_clump':[0.02, 0.05, 1.0, 2.0, 4.0],
        'beta_clump':[0.02, 0.05, 1.0, 2.0, 4.0],
        'gamma_clump':[1.0, 2.0, 4.0, 8.0, 16.0]
    }


    for param in params :

        param_ind = param_ind_dict[param]
        param_val_list = param_values[param]
        color_list = ['C0', 'C1', 'C', 'C3', 'C4']

        f = plt.figure( figsize=(4,4) )
        ax = f.add_axes([0.21,0.16,0.75,0.75])

        ax.errorbar(rosat_ell, rosat_cl, yerr = rosat_cl_err, color='k', fmt='o', label=r"observed", markersize = 5)

        #ax.errorbar(ell_k, cl_k, yerr = cl_k_err, label=r'Kolodzig+18', color='r', fmt='+', markersize=5)
        cl_list = []
        for counter ,param_val in enumerate(param_values[param]) :
            theta = theta_fid.copy()
            theta[param_ind] = param_val

            print(theta[param_ind])
            start = time.time()
            cl = power (ell, theta)
            end = time.time()
            print("Elapsed time: %s" % (end - start))
            cl *= ell*(ell+1)/(2.0*math.pi)
            #psn = psn*ell*(ell+1)/(2.0*math.pi)
            #cl_total *= ell*(ell+1)/(2.0*math.pi)
            cl_list.append(cl)

            label_str = param_label_dict[param]+r'$= %.3f $'% (param_val)
            #if param == 'eps_f' :
            #    label_str = param_label_dict[param]+r'$= %.1f$'% (param_val)
            #if param == 'clump0' :
            #    label_str = param_label_dict[param]+r'$= %.1f$'% (param_val+1)
            #ax.plot (ell, cl_total, ls = '-', color=color_list[counter], label=label_str)
            ax.plot (ell, cl, ls = '-', color=color_list[counter], label=label_str)
            #ax.plot (ell, psn, ls = ':', color=color_list[counter])

        ax.set_xlim ( 10, 1e4 )
        #ax.set_ylim ( 1e-19, 1e-14)
        #ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$\ell(\ell+1)C_{\ell}^{xx}/2\pi\,[{\rm erg^{2}s^{-2}cm^{-4}sr^{-2}}]$')
        #ax.set_ylabel(r'$C_{\ell}\,[{\rm erg^{2}s^{-2}cm^{-4}sr^{-2}}]$')
        #ax.set_ylabel(r'$\ell(\ell+1)C_{\ell}^{xx}/2\pi\,[{\rm keV^{2}s^{-2}cm^{-4}sr^{-2}}]$')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='upper left', prop={'size': 10})

        #outname = '../plots/'+param+'_xx_power.pdf'
        outname = param+'_xx_power.png'
        f.savefig(outname)
        f.clf()
    
        #print (cxb(theta_fid))

if __name__ == "__main__" :
    main()
