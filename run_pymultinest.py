#!/usr/bin/env python
# coding: utf-8

import math
import os, sys, time, logging
import numpy as np
import pandas as pd
import xx_power
import datetime
import pymultinest
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0 :
    now = datetime.datetime.now()
    dirname = "../halo_model_Flender/MultiNest/{0:%Y-%m-%d}".format(now)
    if os.path.exists(dirname) == False:
        os.mkdir(dirname)
    else:
        print("Warning: directory %s already exists" % dirname)
        flag = True
        i = 1
        while flag:
            dirname_try = dirname + "_%s" % i
            i += 1
            if os.path.exists(dirname_try):
                print("Warning: directory %s already exists" % dirname_try)
            else:
                flag = False
                dirname = dirname_try
                os.mkdir(dirname)
                #config["output"]["directory"] = dirname
    print("output directory: %s" % dirname)
else:
    dirname = None #comm.recv(source=0, tag=11)
dirname = comm.bcast(dirname, root=0)

# setup logger
filename_log = "mcmc_log"
logging.basicConfig(filename=os.path.join(dirname, filename_log), level=logging.DEBUG)

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

if rank == 0 :
    ell,cl,var = read_data('../ROSAT/rosat_R4_R7_mask_hfi_R2_small_ell.txt')

    icov = np.zeros((var.size,var.size))
    for i in range(var.size) :
        icov[i,i] = 1.0/var[i]

else :
    ell = None
    cl = None
    icov = None

ell = comm.bcast(ell, root = 0)
cl = comm.bcast(cl, root = 0)
icov = comm.bcast(icov, root = 0)

def power_model (x, params) :

    # set cosmology and linear power spectrum
    H0=70.000000
    Omega_M=0.279000
    Omega_b=0.046100
    w0=-1.000000
    Omega_k=0.000000
    n_s=0.972000
    nH = 1.e22
    inputPk="../input_pk/wmap9_fid_matterpower_z0.dat"
    xx_power.init_cosmology(H0, Omega_M, Omega_b, w0, Omega_k, n_s, nH, inputPk)

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
    eps_f, f_star, S_star, clump0, noise = params[0], params[1], params[2], params[3], param[4]

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
    clump_zslope = 0.0
    x_clump = 1.23
    alpha_clump1 = 0.88
    alpha_clump2 = 3.85

    xx_power.set_Flender_params(alpha0, n_nt, beta, eps_f*1e-6, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod, clump0, clump_zslope, x_clump, alpha_clump1, alpha_clump2, noise)

    model = xx_power.return_xx_power(x) # [erg cm^-2 s^-1 str^-1]^2

    return model

def prior(cube, ndim, nparams): 
    cube[0] = 10.**(cube[0]*1.0 - 2.0) # 0.1 <= eps_f <= 10.0
    cube[1] = cube[1]*(0.032 - 0.020) + 0.020 # 0.020 <= f_star <= 0.032
    cube[2] = 10.**(cube[2]*0.0 - 4.0)  # 0.0001 <= S_star <= 1.0
    cube[3] = cube[3]*(2.0 - 0.0) + 0.0 # 0.0 <= clump0 <= 2.0 
    #cube[4] = cube[4] # -1.0 <= clump_zslope <= 1.0 
    cube[4] = 10.**(cube[4]*(-20.0 - 3.0))

    return cube
 
def loglike(cube, ndim, nparams):

    model = power_model(ell,cube)
    diff = np.array(cl-model, dtype=np.float64)
    lnl = -0.5*np.dot(diff, np.dot(icov, np.transpose(diff)))
    return lnl

#initial paramaters for MCMC

param_list = ['eps_f','f_star', 'S_star', 'clump0','noise']
param_labels = [r'$\epsilon_f$',r'$f_\star$', r'$S_\star$', r'$C_0$', r'$P_{\rm SN}$']

n_params = len(param_list)
nlive = 2048 # number of live points
tol = 0.5   # stopping criterion

# run MultiNest
prefix = dirname+'/test_'
nbunch = 1
start = time.time()

for i in range(nbunch):
    if i == 0 :
        result = pymultinest.run(loglike, prior, n_params, outputfiles_basename=prefix, resume = False, verbose = True, evidence_tolerance = tol, n_live_points=nlive) 
    else :
        result = pymultinest.run(loglike, prior, n_params, outputfiles_basename=prefix, resume = True, verbose = True, evidence_tolerance = tol, n_live_points=nlive) 

    logging.info("%s/%s bunch completed." % (i+1, nbunch))
    end = time.time()
    logging.info("Elapsed time: %s" % (end - start))



