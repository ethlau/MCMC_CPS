#!/usr/bin/env python
# coding: utf-8

import math
import os, sys, time, logging
import numpy as np
import pandas as pd
import xx_power
import datetime
from scipy.optimize import curve_fit

flux_lim = 3.86e-13

# set cosmology and linear power spectrum
H0=70.000000
Omega_M=0.279000
Omega_b=0.046100
w0=-1.000000
Omega_k=0.000000
n_s=0.972000
inputPk="../input_pk/wmap9_fid_matterpower_z0.dat"
xx_power.init_cosmology(H0, Omega_M, Omega_b, w0, Omega_k, n_s, inputPk)

def power_spectrum(x, *theta):
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
    #alpha0, n_nt, beta, eps_f, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod = theta
    #xx_power.set_Flender_params(alpha0, n_nt, beta, eps_f*1e-6, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod)

    #eps_f, f_star, S_star, gamma_mod0, gamma_mod_zslope, clump0, clump_zslope = theta
    eps_f, f_star, S_star, clump0, clump_zslope = theta
    #fix DM profile
    eps_DM = 0.006
    A_C = 1.0

    #fix non-thermal pressure term
    alpha0 = 0.18
    n_nt = 0.80
    beta = 0.50
    x_smooth = 0.01
    n_nt_mod = 0.80
    x_break = 0.195

    gamma_mod0 = 0.10
    gamma_mod_zslope = 1.72

    #clumping terms
    #clump0 = 0.0
    #clump_zslope = 0.0
    x_clump = 1.23
    alpha_clump1 = 0.88
    alpha_clump2 = 3.85

    xx_power.set_Flender_params(alpha0, n_nt, beta, eps_f*1e-6, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod, clump0, clump_zslope, x_clump, alpha_clump1, alpha_clump2)

    model = xx_power.return_xx_power(x,flux_lim) # [erg cm^-2 s^-1 str^-1]^2

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

ell,cl,var = read_data('../ROSAT/rosat_R4_R7_mask_hfi_R2_small_ell.txt')

err = np.sqrt(var)

icov = np.zeros((var.size,var.size))
for i in range(var.size) :
    icov[i,i] = 1.0/var[i]

def lnprob_global(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, ell, cl, icov)

#initial paramaters for MCMC
#pinit  = np.array([1.0,0.0050000,0.026000,0.120000,1.000000,0.100000,1.720000])
pinit  = np.array([5.0,0.026000,0.120000,0.67,0.1])
p0 = pinit.tolist()
print (p0)
ndim = pinit.size

popt, pcov = curve_fit(power_spectrum, ell, cl, p0=p0)

print (popt, pcov)
