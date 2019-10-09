#!/usr/bin/env python
# coding: utf-8

import math
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import corner

'''
argvs = sys.argv
argc = len(argvs)
if (argc != 3):
    print('Usage: # python %s indir output' % argvs[0])
    quit()

dirname=str(argvs[1])
ofname=str(argvs[2])
'''
#dirname="/home/ethlau/projects/Power_Spectrum/halo_model_Flender/MCMC/test/2019-08-19"
dirname="/home/ethlau/projects/Power_Spectrum/halo_model_Flender/MCMC/log/2019-09-04"
ofname='test.pdf'
filename_backend = os.path.join(dirname, "backend.h5")

ndim = 8

reader = emcee.backends.HDFBackend(filename_backend,read_only=True)
full_samples = reader.get_chain()
print (full_samples.shape)

#eps_f, f_star, S_star, gamma_mod0, gamma_mod_zslope, clump0, clump_zslope, log_noise
#labels=[r"$10^{6}\epsilon_{f}$", r"$f_{*}$", r"$S_{*}$", r"$\Gamma_0$", r"$\beta_{\Gamma}$", r"$C_0$",r"$\beta_{C}$", r"$\log P_{\rm SN}$"]
#labels=[r"$10^{6}\epsilon_{f}$", r"$\epsilon_{DM}$", r"$f_{*}$", r"$S_{*}$", r"$C_0$"]
#labels=[r"$\log_{10}(10^{6}\epsilon_{f})$", r"$\log_{10}f_{*}$", r"$\log_{10}(C_0-1)$"]
#labels=[r"$10^{6}\epsilon_{f}$", r"$f_{*}$", r"$S_{*}$", r"$C_0-1$", r"$\alpha_C$"]
#labels=[r"$\log_{10}(10^{6}\epsilon_{f})$", r"$\log_{10}(f_{*})$", r"$\log_{10}(S_{*})$", r"$\log_{10}(C_0-1)$", r"$\log_{10}(\alpha_C)$", r"$\log_{10}(\beta_C)$", r"$\log_{10}(\gamma_C)$"]
labels=[r"$(10^{6}\epsilon_{f})$", r"$f_{*}$", r"$S_{*}$", r"$\Gamma_0$", r"$(C_0-1)$", r"$\alpha_C$", r"$\beta_C$", r"$\gamma_C$"]
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot((full_samples[:, :, i]), "k", alpha=0.3)
    ax.set_xlim(0,  len(full_samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

fig.savefig("chains_log.png")

#tau = reader.get_autocorr_time()
#burnin = int(2*np.max(tau))
#thin = int(0.5*np.min(tau))
burnin=1000
thin=1
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
#log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

print(log_prob_samples)
#print(log_prior_samples)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
#print("flat log prior shape: {0}".format(log_prior_samples.shape))

all_samples = np.concatenate((
    samples, log_prob_samples[:, None]), axis=1)

print (all_samples)

lim_min = np.zeros(ndim)
lim_max = np.zeros(ndim)

for i in np.arange(ndim) :

    #lim_min[i] = np.min(10**samples[:,i])
    #lim_max[i] = np.max(10**samples[:,i])
    lim_min[i] = np.min(samples[:,i])
    lim_max[i] = np.max(samples[:,i])


#lim_min[0] = 8.78
#lim_min[1] = 0.0289
#lim_max[3] = 0.013

#labels += ["log prob", "log prior"]
#fig_c = corner.corner(samples, labels=labels)
#fig_c = corner.corner(10**samples,labels=labels, weights=None, levels=[1-np.exp(-0.5),1-np.exp(-0.5*4.0)], show_titles=True, plot_datapoints=False, title_kwargs={"fontsize": 14}, smooth=None, range=[(1.0,10.0),(0.06,0.1),(0.0,5.0)], quantiles=[0.16, 0.5, 0.84]) 
#fig_c = corner.corner(samples,labels=labels, weights=None, levels=[1-np.exp(-0.5),1-np.exp(-0.5*4.0)], show_titles=False, plot_datapoints=False, title_kwargs={"fontsize": 14}, smooth=None, range=[(lim_min[0], lim_max[0]), (lim_min[1], lim_max[1]),(lim_min[2], lim_max[2]),(lim_min[3], lim_max[3]),(lim_min[4], lim_max[4])] , use_math_text=True)

#fig_c = corner.corner(samples, show_titles=True, plot_datapoints=False)
#range=[(8.0, 9.0),(0.002, 0.0030),(0.02, 0.22), (0.0001, 0.014), (2.50, 3.00)]) 
fig_c = corner.corner(10**samples,labels=labels, weights=None, quantiles=[0.16, 0.50, 0.84], levels=[0.68, 0.95], show_titles=True, plot_datapoints=False, title_kwargs={"fontsize": 14}, smooth=1.0)
fig_c.savefig("corner_log.png")

for i in range(ndim):
    mcmc = np.percentile(10**samples[:, i], [0.02,15.6,50,84.1,97.7])
    #q = np.diff(mcmc)
    q = mcmc - mcmc[2]
    #print(mcmc[2], q[0], q[1], q[2], q[3], labels[i])
    print(mcmc, q, labels[i])

quit()

