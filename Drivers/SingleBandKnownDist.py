import numpy as np
import matplotlib.pyplot as plt

from SMPyBandits.Distribution import *
from SMPyBandits.Environment import *
from SMPyBandits.Policies import HMABPolicy
from SMPyBandits.Policies import HMABAlgo4
from SMPyBandits.Environment import Evaluator

"""
Testing scenario where there is a single band with N (very large, approximately infinite) bins
We know the Band Distribution
But we don't know the Bin Distribution
"""

nbBins = 1000
nbBands = 3
percentile = .99 # p i.e. set percentage of environment that we want the chosen bin to be better than
HORIZON = 1000000
confidence = 0.95 #
t =1

# Gaussian Setup #
