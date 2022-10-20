"""
This script starts a single shell simulation with given parameters.

"""

from time import time
import numpy as np
import dedalus.public as d3
from dedalus.core.operators import SphereEllProduct
from d3shell import d3shellsolver
from convert2matlab import save_to_matlab
import logging
logger = logging.getLogger(__name__)


# Simulation units
micron = 1
joule = 1e16 # we set 1 simulation energy unit to be x pN micrometer
#joule = 1e16 # we set 1 simulation energy unit to be x pN micrometer
minute = 1e3  # simulation unit = 2 second.

# Parameters
#Nphi = 256
#Ntheta = 128

Nphi = 128
Ntheta = 64

# for parallization try 16 cores (or 32)


poisson = 0.3
h = 100 * 1e-3 * micron
R = 15 * micron
FvK = 0*1e2
#FvK = 12 * (1 - poisson**2) * ( R / h) **2
print("FvK = ", FvK)
kT = 2* 4.11e-21 * joule


kappa =  1e-19 * joule
kappa = 10* 1e-19 * joule
Young2d = FvK*kappa / R**2
pc = 4*np.sqrt(kappa * Young2d) / R**2
pressureR =  0.1 * R * pc
pressureR = - 1e2 * kappa *R
#pressureR = 0.5 * R *  2* np.sqrt(kappa*1e4)
lameL = Young2d * poisson / ((1 - poisson )*(1-2*poisson))
lamemu = Young2d / (3*(1 + poisson ))
visco = (4*1.6e-20 * joule / micron**3 * minute)  # 1 Pa.s

params = {'kappa':kappa, 'young':Young2d, 'kT':kT, 'p':pressureR/R, 'R':R, 'poisson':0.3, 'visco':visco}
resolution = (Nphi, Ntheta)
chunk = 20
output_filename = 'outd3_test1.mat'
timestep = 0.1
stop_sim_time = 20 * minute


d3shellsolver(params, output_filename, resolution, chunk, timestep, stop_sim_time)

