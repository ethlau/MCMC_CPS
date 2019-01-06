#!/usr/bin/env python
# coding: utf-8

import math
import os, sys, time, logging
import numpy as np
import pandas as pd
import xx_power
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
    #dirname = "../halo_model_Flender/MCMC/test/{0:%Y-%m-%d}".format(now)
    dirname = "../halo_model_Flender/MCMC/test/2019-01-03_11"
    if os.path.exists(dirname) == False:
        os.mkdir(dirname)
    else:
        print("Warning: directory %s already exists" % dirname)
        flag = False
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
#filename_log = "mcmc_log"
#logging.basicConfig(filename=os.path.join(dirname, filename_log), level=logging.DEBUG)

# set cosmology and linear power spectrum
H0=70.000000
Omega_M=0.279000
Omega_b=0.046100
w0=-1.000000
Omega_k=0.000000
n_s=0.972000
inputPk="../input_pk/wmap9_fid_matterpower_z0.dat"
nH = 2.45e20
xx_power.init_cosmology(H0, Omega_M, Omega_b, w0, Omega_k, n_s, nH, inputPk)

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
    #xx_power.set_Flender_params(alpha0, n_nt, beta, eps_f*1e-6, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod)

    #eps_f, f_star, S_star, gamma_mod0, gamma_mod_zslope, clump0, clump_zslope = theta
    eps_f, f_star, S_star, clump0, log_noise  = theta

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

    xx_power.set_Flender_params(alpha0, n_nt, beta, eps_f*1e-6, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod, clump0, clump_zslope, x_clump, alpha_clump1, alpha_clump2 )

    model = xx_power.return_xx_power(x) # [erg cm^-2 s^-1 str^-1]^2
    sn = np.full(x.shape, 10.0**log_noise, dtype = np.float64)
    model += sn

    diff = np.array(y-model, dtype=np.float64)
    lnl = -0.5*np.dot(diff, np.dot(invcov, np.transpose(diff)))
    return lnl

def lnprior(theta):
    #eps_f, f_star, S_star, gamma_mod0, gamma_mod_zslope, clump0, clump_zslope = theta
    eps_f, f_star, S_star, clump0, log_noise = theta

    # see https://arxiv.org/pdf/1610.08029.pdf
    #if 0.1 <= eps_f <= 10.0 and 0.0 <= eps_DM <= 0.10 and 0.020 <= f_star <= 0.032 and 0.01 <= S_star <= 1.0 and 0.1 <= A_C <= 3.0 and 0.01 <= gamma_mod0 <= 0.30 and 0.10 <= gamma_mod_zslope <= 3.0 :
    #if 0.1 <= eps_f <= 10.0 and 0.020 <= f_star <= 0.032 and 0.01 <= S_star <= 1.0 and 0.01 <= gamma_mod0 <= 0.30 and 0.10 <= gamma_mod_zslope <= 3.0 and 0.0 <= clump0 <= 2.0 and -1.0 <= clump_zslope <= 1.0 :
    if 0.05 <= eps_f <= 15.0 and 0.020 <= f_star <= 0.032 and 0.01 <= S_star <= 1.0 and 0.0 <= clump0 <= 2.0 and -23.0 <= log_noise <= -17.0 : 
        return 0.0
    else:
        return -np.inf

def lnprob(theta, x, y, invcov):
    lp = lnprior(theta)
    ll = lnlike(theta, x, y, icov)
    if not np.isfinite(lp):
        return -np.inf, -np.inf
    if not np.isfinite(ll):
        return lp, -np.inf
    return lp + ll, lp

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
    ell,cl,var = read_data('../ROSAT/rosat_R6_mask_hfi_R2_small_ell.txt')

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
    ll = lnlike(theta, ell, cl, icov)
    if not np.isfinite(lp):
        return -np.inf, -np.inf
    if not np.isfinite(ll):
        return lp, -np.inf
    return lp + ll, lp

#initial paramaters for MCMC
pinit  = np.array([10.0,0.026000,0.120000,0.67,-20.5])
ndim = pinit.size

# chain will be saved every nstep. In total nbunch * nstep samplings.
#nbunch = 25
nstep = 50000

# (total_number_of_cores - 1)*2, this should be equal to ndim x integer

nwalkers = (size-1)*2
#nwalkers = 30

if nwalkers < ndim :
    nwalkers = ndim*2 

coords = np.random.randn(nwalkers, ndim)
pos = [pinit + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]

# run MCMC
with MPIPool() as pool:
    if not pool.is_master() :
        pool.wait()
        sys.exit(0)

    filename_backend = os.path.join(dirname, "backend.h5")
    backend = emcee.backends.HDFBackend(filename_backend)
    #backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_global, pool=pool, backend=backend)

    index = 0
    autocorr = np.empty(nstep)
    old_tau = np.inf

    start = time.time()

    for sample in sampler.sample(pos, iterations=nstep, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau

    end = time.time()
    print("Elapsed time: %s" % (end - start))

