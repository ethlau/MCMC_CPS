
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
Omega_l = 1.0-0.279
Omega_b = 0.0461
h = 0.700
h70 = h/0.700
XH = 0.7600
mmw = 0.588
m_p = 1.67e-24
efc = 1e-11 # convert cts to erg/cm^2 for ROSAT in [0.5,2.0] keV for T=1e7K N_H=2.5e20 cm^-2
rho_crit = 1.878e-29 # h^2 g cm^-3  = 2.775e11 h^2 Msun /Mpc^3 
rho_crit *= h*h

mod_param = {}
mod_param["P13"] = [6.41,  1.81,  0.31, 1.33, 4.13]
mod_param["A10"] = [8.403, 1.177, 0.3081, 1.0510, 5.4905]
mod_param["S16"] = [9.13,1.177, 0.3081, 1.0510,6.13]
mod_param["M14lz"]  = [4.33,2.59, 0.26, 1.63,3.3]
mod_param["M14hz"] = [3.47,2.59, 0.15, 2.27 ,3.48]

model_list = ["P13", "A10", "M14hz", "M14lz", "S16"]
model_label = {"P13": r"Planck13", 
             "A10": r"Arnaud10",
             "M14hz": r"McDonald14 High-z",
             "M14lz": r"McDonald14 low-z",
             "S16": r"Sayers16"
             }


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

    return ysz, m500


def arnaud_lx_m(m500) :

    c = 10**(0.193)
    a0 = 1.e44
    m0 = 3.e14
    alpha = 1.76    

    return c*(m500/m0)**alpha

def arnaud_ysz_m(m500) :

    c = 10**(-4.739)
    a0 = 1.e44
    m0 = 3.e14
    alpha = 1.79 

    return c*(m500/m0)**alpha

def P_gnfw (x,m,z,*p) :
    p0    = p[0]
    c     = p[1]
    gamma = p[2]
    alpha = p[3] 
    beta  = p[4]
    #return p0*(c*x)**(-gamma)*(1+(c*x)**alpha)**(-(beta-gamma)/alpha)*hubble**2*mue/mu/0.175*fb
    pgnfw = p0*(c*x)**(-gamma)*(1+(c*x)**alpha)**(-(beta-gamma)/alpha)
    E = np.sqrt(Omega_m*(1+z)**3+Omega_l)
    Mpiv = 3e14

    pgnfw = 1.65e-3*E**(8./3.)*(m/Mpiv)**(2./3.+0.12)*pgnfw*np.sqrt(h70) # in keV cm^-3
    pgnfw = pgnfw/((2.0+2.0*XH)/(3.0+5.0*XH));
    
    return pgnfw

def pressure_profile (x, mass, redshift, theta) :

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

    pressure = yy_power.return_pressure_profile (x, redshift, mass)

    return pressure

def density_profile (x, mass, redshift, theta) :

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

    density = xx_power.return_density_profile (x, redshift, mass) # in cm^-3
    density = density * mmw * m_p # g cm^-3

    E = np.sqrt(Omega_m*(1+redshift)**3+Omega_l)
    density /= rho_crit*E**2 # \rho/\rho_crit(z)
    
    return density

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

def read_observed_density () :

    filename='../profiles/M17_mean_density.txt'    

    r = []
    rho = []
    rho_err = []
    with open(filename,'r') as f:
        for _ in range(5):
            next(f)
        for line in f:
            cols = line.split('  ')
            r.append(float(cols[0]))
            rho.append(float(cols[1]))
            rho_err.append(float(cols[2]))

    r = np.array(r) # r is r/R500
    rho = 10**np.array(rho) # rho is \rho/\rho_crit(z)
    rhoerr = 10**np.array(rho_err)

    return r, rho, rho_err

def main ():

    # set cosmology and linear power spectrum
    H0=70.0
    Omega_M=0.279000
    Omega_b=0.046100
    w0=-1.000000
    Omega_k=0.000000
    n_s=0.972000
    inputPk="../input_pk/wmap9_fid_matterpower_z0.dat"
    nH = 2.4e21
    opt = 1
    xx_power.init_cosmology(H0, Omega_M, Omega_b, w0, Omega_k, n_s, nH, inputPk, opt)
    yy_power.init_cosmology(H0, Omega_M, Omega_b, w0, Omega_k, n_s, nH, inputPk)

    shot_noise = 0.00

    ell = 10.**np.linspace(np.log10(10.),np.log10(3.e4),31)

    theta_fid = [4.0, 3.e-5 ,0.026000,0.120000,1.000000,0.180000,0.800000,0.500000, 0.10000,1.720000,0.195000,0.010000,0.800000,0.00000,0.730000,1.230000,0.880000, 3.85000]

    param_ind_dict = {'eps_f':0, 'eps_DM':1, 'f_star':2, 'S_star':3, 'A_C':4, 'alpha_nt':5, 'n_nt':6, 'beta_nt':7, 'gamma_mod0':8, 'gamma_mod_zslope':9, 'x_break':10, 'x_smooth':11, 'n_nt_mod':12, 'clump0':13, 'clump_zslope':14, 'x_clump':15, 'alpha_clump1':16, 'alpha_clump2':17}

    param_label_dict = {'eps_f':r'$\epsilon_f$', 'eps_DM':r'$\epsilon_{DM}$', 'f_star':r'$f_\star$', 'S_star':r'$S_\star$', 'A_C':r'$A_C$','alpha_nt':r'$\alpha_{nt}$', 'n_nt':r'$n_{nt}$', 'beta_nt':r'$\beta_{nt}$', 'gamma_mod0':r'$\Gamma_0$', 'gamma_mod_zslope':r'$\beta_\Gamma$', 'n_nt_mod':'$n_{nt,mod}$', 'clump0':r'$C_0$', 'clump_zslope':r'$\beta_C$','x_clump':r'$x_{C}$', 'alpha_clump1':r'$\alpha_{C1}$', 'alpha_clump2':r'$\alpha_{C2}$'}

    #rosat_ell, rosat_cl, rosat_var = read_data("../ROSAT/rosat_R4_R7_mask_hfi_R2_small_ell.txt")
    #rosat_cl *= rosat_ell*(rosat_ell+1.)/(2.0*math.pi)
    #rosat_cl_err = np.sqrt(rosat_var)*rosat_ell*(rosat_ell+1.)/(2.0*math.pi)

    #params = ['eps_f', 'f_star', 'S_star', 'alpha_nt', 'n_nt', 'beta_nt', 'gamma_mod0', 'gamma_mod_zslope', 'clump0', 'clump_zslope', 'x_clump', 'alpha_clump1', 'alpha_clump2' ]
    params = ['eps_f']

    mvir = 3.e13
    #mvir = 10.**np.linspace(13.0, 15.8, 25)
    z = 1.0

    #obs_list = ['mgas', 'lx', 'tx', 'ysz']
    profile_list = ['pressure', 'density']
    profile_label= {
        'pressure': r'$P(r/R_{500})\,{\rm [keV cm^-3]}$',
        'density': r'$\rho_{\rm gas}/\rho_{\rm crit}$'
    }

    x = 10**np.linspace(-3., 0.5, 100)
    ysz, m500 = ysz_m ([mvir], z, theta_fid)
    for param in params :
        pro = {}
        f = {}
        ax = {}

        for o in profile_list :
            f[o] = plt.figure( figsize=(5,5) )
            ax[o]  = f[o].add_axes([0.18,0.16,0.75,0.75])
 
        param_ind = param_ind_dict[param]
        param_fid = theta_fid[param_ind]
        param_val_list = []
        color_list = ['C0', 'C1', 'C2', 'C3', 'C4']
       
        for i in [0.1,0.5,1.0,1.5,2.0]:
        #for i in [1.0]:
            param_val = param_fid * i
            param_val_list.append(param_val)

        cl_list = []
        for counter ,param_val in enumerate(param_val_list) :
            print(param, param_val)
            theta = theta_fid.copy()
            theta[param_ind] = param_val
            pro['pressure'] = pressure_profile(x, mvir, z, theta)
            pro['density'] = density_profile(x, mvir, z, theta)
           
            #print(x, pro['density'])

            label_str = param_label_dict[param]+r'$= %.3f $'% (param_val)
            if param == 'eps_f' :
                label_str = param_label_dict[param]+r'$= %.1f \times 10^{-6}$'% (param_val)
            for o in profile_list :
                ax[o].plot(x, pro[o], label=label_str)

        arnaud_pro = P_gnfw(x,m500,z, *mod_param['A10'])
        robs, rho_obs, rho_obs_err = read_observed_density()
        ax['density'].errorbar(robs, rho_obs, yerr = rho_obs_err,  color='k', label='M17') 
        ax['pressure'].plot(x, arnaud_pro, color='k', label=r'Arnaud+10')

        for o in profile_list :
            ax[o].set_xlabel(r'$r/R_{500c}$')
            ax[o].set_ylabel(profile_label[o])
            ax[o].legend(loc='lower left')
            
            ax[o].set_xscale('log')
            ax[o].set_yscale('log')

            outname = param+'_'+o+'_profile.png'
            f[o].savefig(outname)
            f[o].clf()
    

if __name__ == "__main__" :
    main()
