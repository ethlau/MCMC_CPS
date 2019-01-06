#!/usr/bin/env python
# coding: utf-8

import math
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
argvs = sys.argv
argc = len(argvs)
if (argc != 3):
    print('Usage: # python %s indir output' % argvs[0])
    quit()

dirname=str(argvs[1])
ofname=str(argvs[2])
'''
dirname="/home/ethlau/projects/Power_Spectrum/halo_model_Flender/MCMC/test/2019-01-03_10"
ofname='test.pdf'

# chain will be saved every nstep. In total nbunch * nstep samplings.
nbunch = 2
nstep = 1000

#nwalkers = (size-1)*2 # (total_number_of_cores - 1)*2

# read MCMC information
ndim = 5

try :
    chains_npy="chains_"+str(nbunch-1)+".npy"
    filename_bunch_chains = os.path.join(dirname, chains_npy)
    X = np.load(filename_bunch_chains)

    lnp_npy="lnp_"+str(nbunch-1)+".npy"
    filename_bunch_lnp = os.path.join(dirname, lnp_npy)
    Y = np.load(filename_bunch_lnp)

except IOError:
    print("Files not found!")
    exit(1)

print(X.shape)
print(Y.shape)
#print(Y)
#quit()

labels=[r"$10^{6}\epsilon_{f}$", r"$f_{*}$", r"$S_{*}$", r"$C_0$", r"$\log P{\rm SN}$"]
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(X[:, :, i].T, "k", alpha=0.3)
    ax.set_xlim(0, nstep*nbunch)
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

fig.savefig("chains.png")

# set burn-in
skip = 200

samples = X[:, skip:, :].reshape((-1, ndim))
lnp     = Y[skip:,:].reshape(-1,)

#for i in range(lnp.size):
#    samples[i,0] = samples[i,0] * 1e6
 
min_lnp = lnp.max()
print(lnp, samples.shape, lnp.shape, min_lnp)
weights = np.array(lnp) - np.full(lnp.shape, min_lnp)
weights = np.exp(weights)
print (weights)
norm = weights.sum()
mean = []
for i in range(ndim):
    mean.append(np.dot(samples[:,i], weights))
mean = mean / norm
print(mean)

var = []
for i in range(ndim):
    var.append(np.dot((samples[:,i]-mean[i])*(samples[:,i]-mean[i]), weights))
var = var / norm
print(np.sqrt(var))


import corner
#fig_c = corner.corner(samples, labels=[r"$10^{6}\epsilon_{f}$", r"$\epsilon_{\rm DM}$", r"$f_{*}$", r"$S_{*}$", r"$A_C$", r"$\tilde{\Gamma}$", r"$\gamma$", r"$x_{\rm break}$"],
#                      weights=weights, quantiles=[0.16, 0.50, 0.84], levels=[0.68, 0.95],
#                      range=[(0.1,2.0), (0.00,0.010), (0.020,0.032), (0.01, 0.30), (0.5,1.5), (0.01,0.30), (0.10,3.0), (0.10,0.25)],
 #                     show_titles=True, plot_datapoints=False, title_kwargs={"fontsize": 14}, smooth=1.0)
fig_c = corner.corner(samples,labels=[r"$10^{6}\epsilon_{f}$", r"$f_{*}$", r"$S_{*}$", r"$C_0$", r"$\log P{\rm SN}$"], weights=weights, quantiles=[0.16, 0.50, 0.84], levels=[0.68, 0.95], range=[(0.1,2.0), (0.020,0.032), (0.01, 0.30), (0.01,2.0), (-23,-21)], show_titles=True, plot_datapoints=False, title_kwargs={"fontsize": 14}, smooth=1.0)
fig_c.savefig(ofname)
quit()

# test whether all chain converge to the same target distribution.
# for detail, see e.g.
# http://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introbayes_sect008.htm#statug.introbayes.bayesgelman
# or http://joergdietrich.github.io/emcee-convergence.html

def gelman_rubin(chain):
    ssq = np.var(chain, axis=1, ddof=1)
    W = np.mean(ssq, axis=0)
    thetab = np.mean(chain, axis=1)
    thetabb = np.mean(thetab, axis=0)
    m = chain.shape[0]
    n = chain.shape[1]
    B = n / (m-1) * np.sum((thetabb-thetab)**2, axis=0)
    var_theta = (n-1)/(n) * W + 1./n * B
    Rhat = np.sqrt(var_theta / W)
    return Rhat

chain = X[:, skip:, :]

fig = plt.figure(figsize=(8.9, 5.5))
xmin = 1000
chain_length = chain.shape[1]
step_sampling = np.arange(xmin, chain_length, 50)
for i in range(ndim):
    rhat = np.array([gelman_rubin(chain[:, :steps, :])[i] for steps in step_sampling])
    plt.plot(step_sampling, rhat, label="param{:d}".format(i), linewidth=2)
    
ax = plt.gca()
xmax = ax.get_xlim()[1]
plt.hlines(1.01, xmin, xmax, linestyles="--")
plt.ylabel("$\hat R$")
plt.xlabel("chain length")
plt.ylim(1.00, 1.10)
legend = plt.legend(loc='best')

plt.draw()
plt.pause(20.0)

'''
'''
