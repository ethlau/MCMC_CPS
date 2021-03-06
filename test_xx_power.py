
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

def beam (ell, fwhm=0.5) :

    #convert fwhm from arcmin to radian
    fwhm *= (np.pi/180.0)/60.0
    sigma = fwhm / (np.sqrt(8.0*np.log(2.0)))
    bl = np.exp(ell*(ell+1.0) * sigma**2)

    return bl

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

    print(xx_power.return_total_xsb()/(4.0*math.pi))

    model = xx_power.return_xx_power(ell) # [erg cm^-2 s^-1 str^-1]^2
    #model_2h = xx_power.return_xx_power_2h(ell) # [erg cm^-2 s^-1 str^-1]^2

    return model

def cxb (theta) :

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
    nH = 2.4e21
    opt = 1

    xx_power.init_cosmology(H0, Omega_M, Omega_b, w0, Omega_k, n_s, nH, inputPk, opt)

    shot_noise = 0.00

    #ell = 10.**np.linspace(np.log10(1.),np.log10(1.e5),40)
    ell = np.linspace(1.0, 10000.0, 10000)

    theta_fid = [4.0, 3.e-5 ,0.026000,0.120000,1.000000,0.180000,0.800000,0.500000,0.100000,1.720000,0.195000,0.010000,0.800000,0.670000,0.730000,1.230000,0.880000, 3.85000]

    param_ind_dict = {'eps_f':0, 'eps_DM':1, 'f_star':2, 'S_star':3, 'A_C':4, 'alpha_nt':5, 'n_nt':6, 'beta_nt':7, 'gamma_mod0':8, 'gamma_mod_zslope':9, 'x_break':10, 'x_smooth':11, 'n_nt_mod':12, 'clump0':13, 'clump_zslope':14, 'x_clump':15, 'alpha_clump1':16, 'alpha_clump2':17}

    param_label_dict = {'eps_f':r'$\epsilon_f$', 'eps_DM':r'$\epsilon_{DM}$', 'f_star':r'$f_\star$', 'S_star':r'$S_\star$', 'A_C':r'$A_C$','alpha_nt':r'$\alpha_{nt}$', 'n_nt':r'$n_{nt}$', 'beta_nt':r'$\beta_{nt}$', 'gamma_mod0':r'$\Gamma_0$', 'gamma_mod_zslope':r'$\beta_\Gamma$', 'n_nt_mod':'$n_{nt,mod}$', 'clump0':r'$C_0$', 'clump_zslope':r'$\beta_C$','x_clump':r'$x_{C}$', 'alpha_clump1':r'$\alpha_{C1}$', 'alpha_clump2':r'$\alpha_{C2}$'}

    '''
    rosat_ell, rosat_cl, rosat_var = read_data("../ROSAT/rosat_R4_R7_mask_hfi_R2_small_ell.txt")
    rosat_cl *= rosat_ell*(rosat_ell+1.)/(2.0*math.pi)
    rosat_cl_err = np.sqrt(rosat_var)*rosat_ell*(rosat_ell+1.)/(2.0*math.pi)
    '''
    params = [ 'eps_f', 'f_star', 'S_star', 'alpha_nt', 'n_nt', 'beta_nt', 'gamma_mod0', 'gamma_mod_zslope', 'clump0', 'clump_zslope', 'x_clump', 'alpha_clump1', 'alpha_clump2' ]

    f = plt.figure( figsize=(5,5) )
    ax = f.add_axes([0.18,0.16,0.75,0.75])

    #ax.errorbar(rosat_ell, rosat_cl, yerr = rosat_cl_err, color='k', label=r"ROSAT")

    cl = power (ell, theta_fid)
    nside = 2048
    fwhm = math.radians(12./60.)
    recon_map = hp.sphtfunc.synfast(cl, nside, fwhm=fwhm, pixwin=True, new=True)

    cmap = hp.mollview (recon_map)
    hp.graticule()
    plt.savefig('recon_map.png')

    cl_recon = hp.sphtfunc.anafast(recon_map)
    ell_recon = (np.arange(cl_recon.size)).astype(float)
    print(ell_recon, cl_recon)
    cl *= ell*(ell+1)/(2.0*math.pi)
    cl_recon *= ell_recon*(ell_recon+1)/(2.0*math.pi)

    ax.plot (ell, cl, ls = '-', label = 'original')
    ax.plot (ell_recon, cl_recon, ls = '-', label ='healpix')

    ax.set_xlim ( 10, 3e3 )
    ax.set_ylim ( 1e-19, 1e-15 )
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\ell(\ell+1)C_{\ell}^{xx}/2\pi\,[{\rm erg^{2}s^{-2}cm^{-4}str^{-2}}]$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best')

    #outname = '../plots/'+param+'_xx_power.pdf'
    outname = 'recon_xx_power.png'
    f.savefig(outname)
    f.clf()
    
if __name__ == "__main__" :
    main()
