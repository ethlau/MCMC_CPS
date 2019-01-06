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
dirname="/home/ethlau/projects/Power_Spectrum/halo_model_Flender/MCMC/test/2019-01-03_11"
ofname='test.pdf'
filename_backend = os.path.join(dirname, "backend.h5")

ndim = 5

reader = emcee.backends.HDFBackend(filename_backend,read_only=True)

tau = reader.get_autocorr_time()
burnin = int(2*np.max(tau))
thin = int(0.5*np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat log prior shape: {0}".format(log_prior_samples.shape))

all_samples = np.concatenate((
    samples, log_prob_samples[:, None], log_prior_samples[:, None]
), axis=1)

labels=[r"$10^{6}\epsilon_{f}$", r"$f_{*}$", r"$S_{*}$", r"$C_0$", r"$\log P_{\rm SN}$"]
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, nstep*nbunch)
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

fig.savefig("chains.png")

labels += ["log prob", "log prior"]
fig_c = corner.corner(all_samples, labels=labels);
#fig_c = corner.corner(samples,labels=labels, weights=None, quantiles=[0.16, 0.50, 0.84], levels=[0.68, 0.95], range=[(0.1,2.0), (0.020,0.032), (0.01, 0.30), (0.01,2.0), (-23,-21)], show_titles=True, plot_datapoints=False, title_kwargs={"fontsize": 14}, smooth=1.0)
fig_c.savefig("corner.png")

quit()

