
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

efc = 1e-11 # convert cts to erg/cm^2 for ROSAT in [0.5,2.0] keV for T=1e7K N_H=2.5e20 cm^-2

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

    clump0 = theta[12]
    alpha_clump = theta[13]
    beta_clump = theta[14]
    gamma_clump = theta[15]

    xx_power.set_Flender_params(eps_f*1e-6, eps_DM, f_star, S_star, A_C, A_nt, B_nt, gamma_nt, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, clump0, alpha_clump, beta_clump, gamma_clump)

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

    clump0 = theta[12]
    alpa_clump = theta[13]
    beta_clump = theta[14]
    gamma_clump = theta[15]

    xx_power.set_Flender_params(eps_f*1e-6, eps_DM, f_star, S_star, A_C, A_nt, B_nt, gamma_nt, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, clump0, alpha_clump, beta_clump, gamma_clump)

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

def read_param(filename) :

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
    nH = 0
    opt = 1

    xx_power.init_cosmology(H0, Omega_M, Omega_b, w0, Omega_k, n_s, nH, inputPk, opt)

    shot_noise = 3.0e-21

    ell = 10.**np.linspace(np.log10(10.),np.log10(3.e4),31)

    theta_best = [2.3932, 3.e-5 ,0.0254,0.09520,1.000000,0.45200,0.841000,1.628000,0.102400,1.720000,0.195000,0.010000,1.6960,0.9227,2.4602,4.5937]

    theta_best = np.array(theta_best,dtype='double')

    param_ind_dict = {'eps_f':0, 'eps_DM':1, 'f_star':2, 'S_star':3, 'A_C':4, 'A_nt':5, 'B_nt':6, 'gamma_nt':7, 'gamma_mod0':8, 'gamma_mod_zslope':9, 'x_break':10, 'x_smooth':11, 'clump0':12, 'alpha_clump':13, 'beta_clump':14, 'gamma_clump':15}

    param_label_dict = {'eps_f':r'$\epsilon_f/10^{-6}$', 'eps_DM':r'$\epsilon_{DM}$', 'f_star':r'$f_\star$', 'S_star':r'$S_\star$', 'A_C':r'$A_C$','A_nt':r'$A_{nt}$', 'B_nt':r'$B_{nt}$', 'gamma_nt':r'$\gamma_{nt}$', 'gamma_mod0':r'$\Gamma_0$', 'gamma_mod_zslope':r'$\beta_\Gamma$', 'n_nt_mod':'$n_{nt,mod}$', 'clump0':r'$C_0$', 'clump_zslope':r'$\beta_C$','x_clump':r'$x_{C}$', 'alpha_clump1':r'$\alpha_{C1}$', 'alpha_clump2':r'$\alpha_{C2}$'}

    rosat_ell, rosat_cl, rosat_var = read_data("../ROSAT/rosat_R4_R7_chandra_unabsorbed.txt")
    #rosat_cl *= rosat_ell*(rosat_ell+1.)/(2.0*math.pi)
    rosat_cl_err = np.sqrt(rosat_var)
    #k_k, ell_k, beam_k, cl_k, cl_k_err = read_kolodzig ()
    #cl_k /= beam_k
    #cl_k *= ell_k*(ell_k+1)/(2.0*math.pi)
    

    #params = [ 'eps_f', 'f_star', 'S_star', 'alpha_nt', 'n_nt', 'beta_nt', 'gamma_mod0', 'gamma_mod_zslope', 'clump0', 'clump_zslope', 'x_clump', 'alpha_clump1', 'alpha_clump2' ]


    f = plt.figure( figsize=(4,4) )
    ax = f.add_axes([0.21,0.16,0.75,0.75])

    ax.errorbar(rosat_ell, rosat_cl, yerr = rosat_cl_err, color='k', fmt='o', label=r"ROSAT")
    #ax.errorbar(ell_k, cl_k, yerr = cl_k_err, label=r'Kolodzig+18')
    start = time.time()
    cl = power (ell, theta_best)
    #cl_up = power (ell, theta_cup)
    #cl_dn = power (ell, theta_cdn)
    end = time.time()
    print("Elapsed time: %s" % (end - start))
    with open('best_fit.txt','w') as outf:
        print('# ell C_ell [ergs^2/s^2/cm^4/sr^2]', file=outf)
        for i in np.arange(len(ell)) :
            print(ell[i], cl[i], file=outf)

    #cl *= ell*(ell+1)/(2.0*math.pi)
    ax.plot (ell, cl, ls = '-', label='best fit model')
    ax.set_xlim ( 100,3e4 )
    #ax.set_ylim ( 1e-19, 1e-14)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$C_{\ell}^{xx},[{\rm erg^{2}s^{-2}cm^{-4}sr^{-2}}]$')
    #ax.set_ylabel(r'$\ell(\ell+1)C_{\ell}^{xx}/2\pi\,[{\rm cts^{2}s^{-2}arcmin^{-2}}]$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left')

    #outname = '../plots/'+param+'_xx_power.pdf'
    outname = 'bf_xx_power.png'
    f.savefig(outname)
    f.clf()
    

if __name__ == "__main__" :
    main()
