#!/usr/bin/env python
# coding: utf-8

import math
import os, sys, time, logging
import numpy as np
import pandas as pd
import xx_power
import emcee
import datetime
from emcee.utils import MPIPool
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# setup output directory
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

flux_lim = 3.86e-13

if rank == 0:
    now = datetime.datetime.now()
    dirname = "../halo_model_Flender/MCMC/test/{0:%Y-%m-%d}".format(now)
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
else:
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
xx_power.init_cosmology(H0, Omega_M, Omega_b, w0, Omega_k, n_s, inputPk)

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
    diff = np.array(y-model, dtype=np.float64)
    lnl = -0.5*np.dot(diff, np.dot(invcov, np.transpose(diff)))
    return lnl

def lnprior(theta):
    #eps_f, f_star, S_star, gamma_mod0, gamma_mod_zslope, clump0, clump_zslope = theta
    eps_f, f_star, S_star, clump0, clump_zslope = theta
    # see https://arxiv.org/pdf/1610.08029.pdf
    #if 0.1 <= eps_f <= 10.0 and 0.0 <= eps_DM <= 0.10 and 0.020 <= f_star <= 0.032 and 0.01 <= S_star <= 1.0 and 0.1 <= A_C <= 3.0 and 0.01 <= gamma_mod0 <= 0.30 and 0.10 <= gamma_mod_zslope <= 3.0 :
    #if 0.1 <= eps_f <= 10.0 and 0.020 <= f_star <= 0.032 and 0.01 <= S_star <= 1.0 and 0.01 <= gamma_mod0 <= 0.30 and 0.10 <= gamma_mod_zslope <= 3.0 and 0.0 <= clump0 <= 2.0 and -1.0 <= clump_zslope <= 1.0 :
    if 0.1 <= eps_f <= 10.0 and 0.020 <= f_star <= 0.032 and 0.01 <= S_star <= 1.0 and 0.0 <= clump0 <= 2.0 and -1.0 <= clump_zslope <= 1.0 :
 
        return 0.0
    else:
        return -np.inf

def lnprob(theta, x, y, invcov):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, invcov)


# read data
'''
file="dvec.dat"
df = pd.read_csv(file, delim_whitespace=True, header=None)
df.columns=['ell', 'cl_xx']
x = np.array(df['ell'])
y = np.array(df['cl_xx'])
'''

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
pinit  = np.array([5.0,0.026000,0.120000,0.67,0.0])

# chain will be saved every nstep. In total nbunch * nstep samplings.
nbunch = 5000
nstep = 10
#nwalkers = (size-1)*2 # (total_number_of_cores - 1)*2, this should be equal to ndim x integer

nwalkers = 70  # (total_number_of_cores - 1)*2, this should be equal to ndim x integer
pool = MPIPool(loadbalance=True)
if not pool.is_master():
    pool.wait()
    sys.exit(0)

# run MCMC
ndim = pinit.size
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(ell,cl,icov), pool=pool)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_global, pool=pool)
start = time.time()

for i in range(nbunch):
    if i == 0:
        pos = [pinit + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]
    else :
        pos = sampler.chain[:,-1,:]
    sampler.run_mcmc(pos, nstep)

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

pool.close()


'''
nvec=y.size

file="invcov.dat"
df = pd.read_csv(file, delim_whitespace=True, header=None)
df.columns=['icov']
icov = np.array(df['icov'])
icovtd = icov.reshape(nvec, nvec)
'''

'''
# generate test data
I_E = 1.478585e+01 # X-ray mean intensity [cm^-2 s^-1 str^-1]
A_eff = 500.0      # effective area [cm^2]
T_obs = 4.0        # observational time [year]
sigma_b = 15.0     # beam size [arcsec]
sky_area = 10000.0   # sky coverage [sq.degs]

T_obs = T_obs * 365. * 24. * 60.0 * 60.0  # in second
sigma_b = sigma_b / 60.0 / 60.0 * np.pi /180.0 # in radian
fsky = sky_area / 41252.96

lmin=100
lmax=10000
nbin=10
dlnell = np.log(lmax/lmin)/nbin
lnell = np.arange(start=np.log(lmin), stop=np.log(lmax), step=dlnell , dtype='float64')
ell = np.exp(lnell)

p_data = np.array([0.180000, 0.800000, 0.500000, 1.00, 0.0050000, 0.026000, 0.120000, 1.000000, 0.100000, 1.720000, 0.195000, 0.010000, 0.800000])
s_data = np.array([I_E, A_eff, T_obs, sigma_b, dlnell, fsky])

def data_and_cov(x, theta, theta_n):
    alpha0, n_nt, beta, eps_f, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod = theta
    I_E, A_eff, T_obs, sigma_b, dlnell, fsky = theta_n
    xx_power.set_Flender_params(alpha0, n_nt, beta, eps_f*1e-6, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod, clump0, clump_zslope, x_clump, alpha_clump1, alpha_clump2)
    fac = (41252.96) * (60.0*60.0) * (60.0*60.0) / (4.*np.pi) # arcsec^2/str
    data = xx_power.return_xx_power(x) # [ergs cm ^-2 s^-1 str^-2]^2

    #N_photon = I_E * A_eff * T_obs # photon/str
    #I_E = I_E / fac # cm^-2 s^-1 arcsec^-2
    #data_n = I_E * I_E / N_photon

    icov = np.zeros((x.size, x.size))
    for i in range(x.size):
        window = np.exp(-x[i]*x[i]*sigma_b*sigma_b/2.0)
        Nmode = (2*x[i]+1)*x[i]*dlnell*fsky
        cov = 2.0*(data[i]+data_n/window/window)*(data[i]+data_n/window/window)/Nmode

        # add statistical noise
        mu, sigma = 0.0, np.sqrt(cov)
        stat_n = np.random.normal(mu, sigma, 1)
        data[i] = data[i] + stat_n

        icov[i,i] = 1./cov

    return data, icov

'''

#x = np.array(ell)
#y, icovtd = data_and_cov(x, p_data, s_data)

#if rank == 0:
#    print (x, y)
#    print (icovtd)


