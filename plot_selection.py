
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xx_power
import time
import scipy.interpolate as interpolate

from hmf import cosmo
from hmf import hmf


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


def xray_flux (mass, redshift, theta) :

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
    alpha_clump = theta[14]
    beta_clump = theta[15]
    gamma_clump = theta[16]

    xx_power.set_Flender_params(alpha0, n_nt, beta, eps_f*1e-6, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod, clump0, alpha_clump, beta_clump, gamma_clump)

    flux = xx_power.return_flux(redshift, mass)
    m500 = xx_power.Mvir_to_Mdeltac(redshift, mass, 500.0)
    
    return flux, m500

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

    ell = 10.**np.linspace(np.log10(10.),np.log10(3.e4),31)

    theta_fid = [4.0, 3.e-5 ,0.0800,0.120000,1.000000,0.180000,0.800000,0.500000,0.10000,1.720000,0.195000,0.010000,0.800000, 0.9, 1.0, 6.0, 3.0]

    param_ind_dict = {'eps_f':0, 'eps_DM':1, 'f_star':2, 'S_star':3, 'A_C':4, 'alpha_nt':5, 'n_nt':6, 'beta_nt':7, 'gamma_mod0':8, 'gamma_mod_zslope':9, 'x_break':10, 'x_smooth':11, 'n_nt_mod':12, 'clump0':13, 'clump_zslope':14, 'x_clump':15, 'alpha_clump1':16,}

    param_label_dict = {'eps_f':r'$\epsilon_f/10^{-6}$', 'eps_DM':r'$\epsilon_{DM}$', 'f_star':r'$f_\star$', 'S_star':r'$S_\star$', 'A_C':r'$A_C$','alpha_nt':r'$\alpha_{nt}$', 'n_nt':r'$n_{nt}$', 'beta_nt':r'$\beta_{nt}$', 'gamma_mod0':r'$\Gamma_0$', 'gamma_mod_zslope':r'$\beta_\Gamma$', 'n_nt_mod':'$n_{nt,mod}$', 'clump0':r'$C_0$', 'clump_zslope':r'$\beta_C$','x_clump':r'$x_{C}$', 'alpha_clump1':r'$\alpha_{C1}$', 'alpha_clump2':r'$\alpha_{C2}$'}

    #rosat_ell, rosat_cl, rosat_var = read_data("../ROSAT/rosat_R4_R7_mask_hfi_R2_small_ell.txt")
    #rosat_cl *= rosat_ell*(rosat_ell+1.)/(2.0*math.pi)
    #rosat_cl_err = np.sqrt(rosat_var)*rosat_ell*(rosat_ell+1.)/(2.0*math.pi)

    #params = ['eps_f', 'f_star', 'S_star', 'alpha_nt', 'n_nt', 'beta_nt', 'gamma_mod0', 'gamma_mod_zslope', 'clump0', 'clump_zslope', 'x_clump', 'alpha_clump1', 'alpha_clump2' ]
    params = ['eps_f', 'f_star', 'clump0']

    redshift, dlz = np.linspace(-4, np.log10(3.0), 10, retstep=True)

    redshift = 10**redshift

    print(dlz)

    my_cosmo = cosmo.Cosmology()
    hubble = my_cosmo.cosmo.h
    print (hubble)

    mvir, dlm = np.linspace(13,16,30,retstep=True)
    flux, dlf = np.linspace(-22,-10,30,retstep=True)

    mvir = 10**mvir
    flux = 10**flux

    Nsperster = np.zeros(flux.shape)

    redo = False

    if redo :
        for iv, f in enumerate(flux):
            for iz, z in enumerate(redshift) :
                dVdz = my_cosmo.cosmo.differential_comoving_volume(z).value*hubble**3
                s = []
                for mass in mvir :
                    ff, m500 = xray_flux ( mass, z, theta_fid )  
                    s.append(ff)
                s = np.array(s)
                mlim = np.interp(f, s, mvir)

                #h = hmf.MassFunction(z=z, Mmin=13, Mmax=16, dlog10m=0.1)
                h = hmf.MassFunction(z=z)
           
                Nm = 0.0 
                for im, mass in enumerate(h.m) :
                    if mass >= mlim :
                        Nm += h.dlog10m*h.dndlog10m[im]*dVdz*z*dlz*(1.+z)**3
                Nsperster[iv] += Nm

        np.save("logNlogS.npy",Nsperster)

    else :
        Nsperster = np.load("logNlogS.npy")
    
    print(Nsperster)

    fig = plt.figure( figsize=(4,4) )
    ax  = fig.add_axes([0.21,0.16,0.75,0.75])
    ax.loglog(flux, Nsperster)
    ax.set_xlabel(r"$S$ [erg/s/cm$^2$]")
    ax.set_ylabel(r"$N(>S)$ [str$^{-1}$]")
    fig.savefig("logNlogS.png")
    fig.clf()

    #print(Nsperster)
    # number per sq deg
    Nspersqdeg = Nsperster/ster2sqdeg

    surveys = ["CDFS", "COSMOS", "XXL", "S82", "ROSAT"]
    #areas = {"CDFS":0.25, "COSMOS":2.0, "XXL":50.0, "S82":31} 
    #sens = {"CDFS":0.66e-15, "COSMOS":1.7e-15, "XXL":5e-15, "S82":0.87e-15} 
    
    areas = [0.25, 2.0, 50.0, 31, 4.0*3.141592*ster2sqdeg*0.4]
    sens = [0.66e-15, 1.7e-15, 5e-15, 0.87e-15, 5.6e-13]

    fig = plt.figure( figsize=(4,4) )
    ax  = fig.add_axes([0.21,0.16,0.75,0.75])
    nhalo_array = np.array([10, 100, 1000])

    area_array = np.linspace(1e-5, 4.0*3.141592, 10000)*ster2sqdeg

    '''
    tot_N = np.zeros( [len(area_array), len(flux)] )
    
    for ix, area in enumerate(area_array) :
        for iv, v in enumerate(flux) :
            tot_N[ix,iv] = area*Nspersqdeg[iv] 

    for nhalo in nhalo_array :
        y = np.zeros(area_array.shape)
        for ix, x in enumerate(area_array) :
            temp_N = tot_N[ix,:]
            y[ix] = 10**np.interp(np.log10(nhalo), np.log10(np.flip(temp_N)),np.log10( np.flip(flux)))


        label_str = r"$N_{\rm cl}>"+str(nhalo)+"$"
        ax.plot(area_array,y,label=label_str)
    '''
    ax.scatter(areas, sens, s=16.0, marker='o')

    for i, txt in enumerate(surveys):
        ax.annotate(txt, (areas[i], sens[i]), xytext = (areas[i], sens[i]*1.25), horizontalalignment='center', size=12)

    
    ax.set_xlabel(r'Area [${\rm deg}^2$]')
    ax.set_ylabel(r'flux [${\rm erg/s/cm^2}$]')
            
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-2, 3e5 )
    ax.set_ylim(1e-16, 3e-12)

    #ax.legend(loc='best', fontsize=10)

    fig.savefig("wedding.png")
    fig.clf() 

if __name__ == "__main__" :
    main()
