#!/usr/bin/env python
# coding: utf-8

import math
import os, sys, time, logging
import numpy as np
import pandas as pd
import xy_power
import emcee
import datetime
from schwimmbad import MPIPool
from mpi4py import MPI
import cProfile

#from multiprocessing import Pool

os.environ["OMP_NUM_THREADS"] = "1"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0 :
    now = datetime.datetime.now()
    dirname = "../halo_model_Flender/MCMC_xy/test/{0:%Y-%m-%d}".format(now)
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
    #comm.send(dirname, dest=1, tag=11)

else :
    dirname = None #comm.recv(source=0, tag=11)

dirname = comm.bcast(dirname, root=0)

# setup logger
filename_log = "mcmc_log"
logging.basicConfig(filename=os.path.join(dirname, filename_log), level=logging.DEBUG)

# set cosmology and linear power spectrum
H0=70.000000
Omega_M=0.279000
Omega_b=0.046100
w0=-1.000000
Omega_k=0.000000
n_s=0.972000
inputPk="../input_pk/wmap9_fid_matterpower_z0.dat"
nH = 2.45e20
xy_power.init_cosmology(H0, Omega_M, Omega_b, w0, Omega_k, n_s, nH, inputPk)

def lnlike(theta, x, y, invcov):
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
    #xy_power.set_Flender_params(alpha0, n_nt, beta, eps_f*1e-6, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod)

    alpha0, n_nt, beta, eps_f, f_star, S_star, clump0 = theta

    #fix DM profile
    eps_DM = 0.006
    A_C = 1.0

    #fix non-thermal pressure term
    #alpha0 = 0.18
    #n_nt = 0.80
    #beta = 0.50
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

    xy_power.set_Flender_params(alpha0, n_nt, beta, eps_f*1e-6, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod, clump0, clump_zslope, x_clump, alpha_clump1, alpha_clump2 )

    model = xy_power.return_xy_power(x) # [erg cm^-2 s^-1 str^-1]^2
    #sn = np.full(x.shape, 10.0**log_noise, dtype = np.float64)
    #model += sn

    diff = np.array(y-model, dtype=np.float64)
    lnl = -0.5*np.dot(diff, np.dot(invcov, np.transpose(diff)))
    return lnl

def lnprior(theta):
    #eps_f, f_star, S_star, gamma_mod0, gamma_mod_zslope, clump0, clump_zslope = theta
    alpha0, n_nt, beta, eps_f, f_star, S_star, clump0 = theta

    # see https://arxiv.org/pdf/1610.08029.pdf
    #if 0.1 <= eps_f <= 10.0 and 0.0 <= eps_DM <= 0.10 and 0.020 <= f_star <= 0.032 and 0.01 <= S_star <= 1.0 and 0.1 <= A_C <= 3.0 and 0.01 <= gamma_mod0 <= 0.30 and 0.10 <= gamma_mod_zslope <= 3.0 :
    #if 0.1 <= eps_f <= 10.0 and 0.020 <= f_star <= 0.032 and 0.01 <= S_star <= 1.0 and 0.01 <= gamma_mod0 <= 0.30 and 0.10 <= gamma_mod_zslope <= 3.0 and 0.0 <= clump0 <= 2.0 and -1.0 <= clump_zslope <= 1.0 :
    if 0.1 <= alpha0 <= 0.3 and 0.5 <= n_nt <= 1.0 and 0.1 <= beta <= 1.0 and 0.1 <= eps_f <= 10.0 and 0.020 <= f_star <= 0.032 and 0.01 <= S_star <= 1.0 and 0.0 <= clump0 <= 2.0 : 
        return 0.0
    else:
        return -np.inf

def lnprob(theta, x, y, invcov):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, invcov)


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
    ell,cl,var = read_data('../ROSAT/rosat_R6_planck_mask_hfi_R2_small_ell.txt')

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

def lnprob_global(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, ell, cl, icov)

#initial paramaters for MCMC
pinit  = np.array([0.18, 0.80, 0.5, 5.0, 0.026000,0.120000,0.67])
ndim = pinit.size

# chain will be saved every nstep. In total nbunch * nstep samplings.
nbunch = 25
nstep = 1000

# (total_number_of_cores - 1)*2, this should be equal to ndim x integer

nwalkers = (size-1)*2
#nwalkers = 30
if rank == 0 :
    print("nwalkers = ", nwalkers)

if nwalkers < ndim :
    nwalkers = ndim*2 

# run MCMC
with MPIPool() as pool:
    if not pool.is_master() :
        pool.wait()
        sys.exit(0)

    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(ell,cl,icov), pool=pool)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_global, pool=pool)
    start = time.time()

    for i in range(nbunch):
        if i == 0:
            pos = [pinit + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]
        else :
            pos = sampler.chain[:,-1,:]
        sampler.run_mcmc(pos, nstep, progress=True)

        chains="chains_"+str(i)
        filename_bunch_chains = os.path.join(dirname, chains)
        np.save(filename_bunch_chains, sampler.chain)
        logging.info("%s/%s bunch completed. File written in %s" % (i+1, nbunch, filename_bunch_chains))

        lnp="lnp_"+str(i)
        filename_bunch_lnp = os.path.join(dirname, lnp)
        np.save(filename_bunch_lnp, sampler.lnprobability)
        logging.info("%s/%s bunch completed. File written in %s" % (i+1, nbunch, filename_bunch_lnp))

        end = time.time()
        logging.info("Elapsed time: %s" % (end - start))

        print("%s/%s bunch completed." % (i+1, nbunch))
        print("Elapsed time: %s" % (end - start))

