
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xx_power, yy_power
import time

keV2erg = 1.6022e-9
deg2rad = math.pi/180.0
ster2sqdeg = 3282.80635
ster2sqarcmin = ster2sqdeg * 3600.0
ster2sqarcsec = ster2sqdeg * 3600.0 * 3600.0
Omega_m = 0.279
Omega_l = 1-Omega_m
Omega_b = 0.0461
f_baryon = Omega_b/Omega_m
h = 0.7
h70 = h/0.7
Mpiv = 3e14
XH = 0.76

efc = 1e-11 # convert cts to erg/cm^2 for ROSAT in [0.5,2.0] keV for T=1e7K N_H=2.5e20 cm^-2

def beam (ell, fwhm=0.5) :

    #convert fwhm from arcmin to radian
    fwhm *= (np.pi/180.0)/60.0
    sigma = fwhm / (np.sqrt(8.0*np.log(2.0)))
    bl = np.exp(ell*(ell+1.0) * sigma**2)

    return bl

def mgas_m (mass, redshift, theta) :

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

    mgas = []
    m500 = []
    for m in mass :
        mgas.append(xx_power.return_Mgas(redshift, m))
        m500.append(xx_power.Mvir_to_Mdeltac(redshift, m, 500.0))
    
    mgas = np.array(mgas)
    m500 = np.array(m500)

    return mgas, m500

def tx_m (mass, redshift, theta) :

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

    tx = []
    m500 = []
    for m in mass :
        tx.append(xx_power.return_Tx(redshift, m))
        m500.append(xx_power.Mvir_to_Mdeltac(redshift, m, 500.0))
    
    tx = np.array(tx)
    m500 = np.array(m500)

    return tx, m500


def lx_m (mass, redshift, theta) :

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

    lx = []
    m500 = []
    for m in mass :
        lx.append(xx_power.return_Lx(redshift, m))
        m500.append(xx_power.Mvir_to_Mdeltac(redshift, m, 500.0))
    
    lx = np.array(lx)/1.e44
    m500 = np.array(m500)

    return lx, m500

def ysz_m (mass, redshift, theta) :

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

    yy_power.set_Flender_params(alpha0, n_nt, beta, eps_f*1e-6, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod, clump0, clump_zslope, x_clump, alpha_clump1, alpha_clump2)
    xx_power.set_Flender_params(alpha0, n_nt, beta, eps_f*1e-6, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod, clump0, clump_zslope, x_clump, alpha_clump1, alpha_clump2)

    ysz = []
    m500 = []
    for m in mass :
        ysz.append(yy_power.return_Ysz(redshift,m))
        m500.append(xx_power.Mvir_to_Mdeltac(redshift, m, 500.0))
    
    ysz = np.array(ysz)
    m500 = np.array(m500)

    E = np.sqrt(Omega_m*(1+redshift)**3+Omega_l)
    ysz = ysz * E**(-2./3.)

    return ysz, m500


def arnaud_lx_m(m500) :

    c = 10**(0.193)
    m0 = 3.e14
    alpha = 1.76    

    return c*(m500/m0)**alpha

def arnaud_ysz_m(m500) :

    c = 10**(-4.739)
    m0 = 3.e14
    alpha = 1.79 

    return c*(m500/m0)**alpha

def mantz_lx_m(m500, z) :

    E = np.sqrt(Omega_m*(1.+z)**3+Omega_l)
    m = np.log(m500/1e15)
    beta0 = 1.70
    beta1 = 1.34
    epsilon = np.log(E)
    gamma = -4.2

    lx = beta0+ beta1*(epsilon+m) - gamma*np.log(h70)
    lx = np.exp(lx)*E

    return lx

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
    yy_power.init_cosmology(H0, Omega_M, Omega_b, w0, Omega_k, n_s, nH, inputPk)

    shot_noise = 0.00

    ell = 10.**np.linspace(np.log10(10.),np.log10(3.e4),31)

    theta_fid = [4.0, 3.e-5 ,0.04500,0.120000,1.000000,0.180000,0.800000,0.500000,0.100000,1.720000,0.195000,0.010000,0.800000,1.00000,0.730000,1.230000,0.880000, 3.85000]

    param_ind_dict = {'eps_f':0, 'eps_DM':1, 'f_star':2, 'S_star':3, 'A_C':4, 'alpha_nt':5, 'n_nt':6, 'beta_nt':7, 'gamma_mod0':8, 'gamma_mod_zslope':9, 'x_break':10, 'x_smooth':11, 'n_nt_mod':12, 'clump0':13, 'clump_zslope':14, 'x_clump':15, 'alpha_clump1':16, 'alpha_clump2':17}

    param_label_dict = {'eps_f':r'$\epsilon_f/10^{-6}$', 'eps_DM':r'$\epsilon_{DM}$', 'f_star':r'$f_\star$', 'S_star':r'$S_\star$', 'A_C':r'$A_C$','alpha_nt':r'$\alpha_{nt}$', 'n_nt':r'$n_{nt}$', 'beta_nt':r'$\beta_{nt}$', 'gamma_mod0':r'$\Gamma_0$', 'gamma_mod_zslope':r'$\beta_\Gamma$', 'n_nt_mod':'$n_{nt,mod}$', 'clump0':r'$C_0$', 'clump_zslope':r'$\beta_C$','x_clump':r'$x_{C}$', 'alpha_clump1':r'$\alpha_{C1}$', 'alpha_clump2':r'$\alpha_{C2}$'}

    #rosat_ell, rosat_cl, rosat_var = read_data("../ROSAT/rosat_R4_R7_mask_hfi_R2_small_ell.txt")
    #rosat_cl *= rosat_ell*(rosat_ell+1.)/(2.0*math.pi)
    #rosat_cl_err = np.sqrt(rosat_var)*rosat_ell*(rosat_ell+1.)/(2.0*math.pi)

    #params = ['eps_f', 'f_star', 'S_star', 'alpha_nt', 'n_nt', 'beta_nt', 'gamma_mod0', 'gamma_mod_zslope', 'clump0', 'clump_zslope', 'x_clump', 'alpha_clump1', 'alpha_clump2' ]
    params = ['eps_f', 'f_star', 'clump0']

    mvir = 10.**np.linspace(13.0, 16.0, 26)
    z = 1e-4

    obs_list = ['mgas', 'lx', 'tx']
    #obs_list = ['mgas']
    obs_label= {
        #'lx': r'$L_X(0.15<r/R_{500}<1,z=0)\,{\rm [10^{44}ergs/s]}$',
        'lx': r'$L_X\,{\rm [10^{44}ergs/s]}$',
        'mgas': r'$M_{\rm gas} (<R_{500}) {[M_\odot]}$',
        'tx': r'$T_X(0.15<r/R_{500}<1,z=0)\,{\rm [keV]}$',
        'ysz': r'$Y_{SZ}(<R_{500},z=0)\,{\rm [Mpc^2] }$'
    }


    for param in params :
        obs = {}
        f = {}
        ax = {}

        for o in obs_list :
            f[o] = plt.figure( figsize=(4,4) )
            ax[o]  = f[o].add_axes([0.21,0.16,0.75,0.75])
 
        param_ind = param_ind_dict[param]
        param_fid = theta_fid[param_ind]
        param_val_list = []
        color_list = ['C0', 'C1', 'C2', 'C3', 'C4']
       
        multi_list =  [0.1,0.5,1.0,2.0]
        #multi_list =  [0.1,0.3,1.0,3.0,10.0]
        if param == 'f_star' :
            multi_list =  [0.5,0.75,1.0,1.25]
        for i in multi_list:

            param_val = param_fid * i
            param_val_list.append(param_val)

        cl_list = []
        for counter ,param_val in enumerate(param_val_list) :
            print(param, param_val)
            theta = theta_fid.copy()
            theta[param_ind] = param_val
            obs['lx'], m500 = lx_m(mvir, z, theta)
            obs['tx'], m500 = tx_m(mvir, z, theta)
            obs['mgas'], m500 = mgas_m(mvir, z, theta)
            #obs['ysz'],m500 = ysz_m(mvir, z, theta)

            label_str = param_label_dict[param]+r'$= %.3f $'% (param_val)
            if param == 'eps_f' :
                label_str = param_label_dict[param]+r'$= %.1f$'% (param_val)
            if param == 'clump0' :
                label_str = param_label_dict[param]+r'$= %.1f$'% (param_val+1)

            for o in obs_list :
                ax[o].plot(m500, obs[o], label=label_str)

        lx_arnaud = arnaud_lx_m (m500)
        lx_mantz = mantz_lx_m (m500,z)
        ysz_arnaud = arnaud_ysz_m (m500)
        #ax['lx'].plot(m500, lx_arnaud, color='k', label=r'Arnaud+10')
        #ax['lx'].plot(m500, lx_mantz, color='k', label=r'Mantz+16')
        #ax['ysz'].plot(m500, ysz_arnaud, color='k', label=r'Arnaud+10')

        for o in obs_list :
            ax[o].set_xlabel(r'$M_{500c}$ [$M_\odot]$')
            ax[o].set_ylabel(obs_label[o])
            ax[o].legend(loc='lower right')
            
            ax[o].set_xscale('log')
            ax[o].set_yscale('log')

            outname = param+'_'+o+'_m_z0.pdf'
            f[o].savefig(outname)
            f[o].clf()
    

if __name__ == "__main__" :
    main()
