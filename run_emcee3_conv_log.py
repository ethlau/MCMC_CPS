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

#os.environ["OMP_NUM_THREADS"] = "1"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0 :
    now = datetime.datetime.now()
    dirname = "../halo_model_Flender/MCMC/log/{0:%Y-%m-%d}".format(now)
    #dirname = "../halo_model_Flender/MCMC/gaussian/{0:%Y-%m-%d}".format(now)
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
filename_log = "mcmc_log"
logging.basicConfig(filename=os.path.join(dirname, filename_log), level=logging.DEBUG)

# set cosmology and linear power spectrum
'''
H0=70.000000
Omega_M=0.279000
Omega_b=0.046100
w0=-1.000000
Omega_k=0.000000
n_s=0.972000
inputPk="../input_pk/wmap9_fid_matterpower_z0.dat"
nH = 2.45e21
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

def beam (ell, fwhm=13.7) :

    #convert fwhm from arcmin to radian
    fwhm = math.radians(fwhm/60.0)
    sigma = fwhm / (np.sqrt(8.0*np.log(2.0)))
    bl = np.exp(ell*(ell+1.0) * sigma**2)

    return bl

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
    '''
    eps_f, f_star, S_star, gamma_mod0, clump0, alpha_clump, beta_clump, gamma_clump = theta
    
    eps_f = 10**eps_f
    f_star = 10**f_star
    S_star = 10**S_star
    gamma_mod0 = 10**gamma_mod0
    clump0 = 10**clump0
    alpha_clump = 10**alpha_clump
    beta_clump = 10**beta_clump
    gamma_clump = 10**gamma_clump

    #fix DM profile
    eps_DM = 3e-5
    A_C = 1.0

    #fix non-thermal pressure term
    A_nt = 0.452
    B_nt = 0.841
    gamma_nt = 1.628
    x_smooth = 0.01
    x_break = 0.195

    #gamma_mod0 = 0.10
    gamma_mod_zslope = 1.72

    #S_star = 0.12

    #clumping terms
    #clump0 = 0.0
    #alpha_clump = 1.0
    #beta_clump = 6.0
    #gamma_clump = 3.0


    xx_power.set_Flender_params(eps_f*1e-6, eps_DM, f_star, S_star, A_C, A_nt, B_nt, gamma_nt, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, clump0, alpha_clump, beta_clump, gamma_clump)
    model = xx_power.return_xx_power_alt(x)
    #sn = np.full(x.shape, 10.0**log_noise, dtype = np.float64)
    #model += sn
    #model /= beam(x)

    diff = np.array(y-model, dtype=np.float64)
    lnl = -0.5*np.dot(diff, np.dot(invcov, np.transpose(diff)))
    return lnl

def lnprior(theta):

    eps_f, f_star, S_star, gamma_mod0, clump0, alpha_clump, beta_clump, gamma_clump = theta

    ## see https://arxiv.org/pdf/1610.08029.pdf
    ## Flat priors on all parameters
    #if 1.09 <= eps_f <= 8.79 and 0.023 <= f_star <= 0.029 and 0.02 <= S_star <= 0.22 and 0.05 <= gamma_mod0 <= 0.21 and  0.01 <= clump0 <= 10.0 and 0.01 <= alpha_clump <= 10.0 and 0.01 <= beta_clump <= 10.0 and 0.01 <= gamma_clump <= 10.0 : 
    if np.log10(1.09) <= eps_f <= np.log10(8.79) and np.log10(0.023) <= f_star <= np.log10(0.029) and np.log10(0.02) <= S_star <= np.log10(0.22) and np.log10(0.05) <= gamma_mod0 <= np.log10(0.21) and  -2 <= clump0 <= 1 and -2 <= alpha_clump <= 1 and -2 <= beta_clump <= 1 and -2 <= gamma_clump <= 1 :
        return 0.0
    else :
        return -np.inf
    
    ## Gaussian priors on non-clumping parameters
    
    #if not 0.01 <= clump0 <= 10.0 and 0.01 <= alpha_clump <= 10.0 and 0.01 <= beta_clump <= 10.0 and 0.01 <= gamma_clump <= 10.0 :
    #    return -np.inf

    ## see https://stackoverflow.com/questions/49810234/using-emcee-with-gaussian-priors
    #eps_m = 3.97
    #eps_s = 4.82/2.0
    #f_star_m = 0.026
    #f_star_s = 0.003/2.0
    #S_star_m = 0.12
    #S_star_s = 0.10/2.0
 
    #gamma_mod0_m = 0.10
    #gamma_mod0_s = 0.11/2.0

    #eps_pr = np.log(1.0/(np.sqrt(2*np.pi)*eps_s))-0.5*(eps_f-eps_m)**2/(eps_s**2)
    #f_star_pr = np.log(1.0/(np.sqrt(2*np.pi)*f_star_s))-0.5*(f_star-f_star_m)**2/(f_star_s**2)
    #S_star_pr = np.log(1.0/(np.sqrt(2*np.pi)*S_star_s))-0.5*(S_star-S_star_m)**2/(S_star_s**2)
    #gamma_mod0_pr = np.log(1.0/(np.sqrt(2*np.pi)*gamma_mod0_s))-0.5*(gamma_mod0-gamma_mod0_m)**2/(gamma_mod0_s**2)
    
    #pr = eps_pr + f_star_pr + S_star_pr + gamma_mod0_pr

    #return pr

def lnprob(theta, x, y, invcov):
    lp = lnprior(theta)
    ll = lnlike(theta, x, y, invcov)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ll

def lnprob_global(theta):
    lp = lnprior(theta)
    ll = lnlike(theta, ell, cl, icov)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ll

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
    #filename = '../ROSAT/rosat_R4_R7.txt'
    filename = '../ROSAT/rosat_R4_R7_chandra_unabsorbed.txt'
    ell,cl,var = read_data(filename)

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

#initial paramaters for MCMC
pinit  = np.log10(np.array([3.97, 0.026, 0.12, 0.10, 1.0, 1.0, 3.0, 3.0]))

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
    backend.reset(nwalkers, ndim)
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

