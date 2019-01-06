#!/usr/bin/env python
# coding: utf-8

import math
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymultinest
import corner

dirname = sys.argv[1]
a = pymultinest.Analyzer(outputfiles_basename=dirname, n_params = 4)

# get a dictionary containing information about
#   the logZ and its errors
#   the individual modes and their parameters
#   quantiles of the parameter posteriors
stats = a.get_stats()

# get the best fit (highest likelihood) point
bestfit_params = a.get_best_fit()

print (bestfit_params)

 
# iterate through the "posterior chain"
#for params in a.get_equal_weighted_posterior():
#    print (params)

