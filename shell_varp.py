"""
This script starts a job array running multiple instances of the langevin shell simulation with  
variable pressure.

Designed to be used with a bash script to send to MIT supercloud's SLURM-based scheduler
see https://supercloud.mit.edu/submitting-jobs; example submission .sh:

______________
#!/bin/bash
 
#SBATCH -o myScript.sh.log-%j-%a
#SBATCH -a 1-4
 
mpiexec -n 4 python shell_varp.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
________________
"""

import sys
from time import time
import numpy as np
from d3shell import d3shellsolver
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

# Grab the arguments that are passed in
# This is the task id and number of tasks that can be used
# to determine which indices this process/task is assigned
my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])

tmpdir = sys.argv[3]



# Simulation units
micron = 1
joule = 1e16 # we set 1 simulation energy unit to be x pN micrometer
#joule = 1e16 # we set 1 simulation energy unit to be x pN micrometer
minute = 1e2  # simulation unit = 2 second.

# Parameters
Nphi = 256
Ntheta = 128

Nphi = 512
Ntheta = 256




poisson = 0.3
h = 100 * 1e-3 * micron
R = 25 * micron
FvK = 3e4
#FvK = 12 * (1 - poisson**2) * ( R / h) **2
#print("FvK = ", FvK)
kT = 2* 4.11e-21 * joule


kappa =  1e-19 * joule 
Young2d = FvK*kappa / R**2
pc = 4*np.sqrt(kappa * Young2d) / R**2
pressure =  0.*-0.8*pc
lameL = Young2d * poisson / ((1 - poisson )*(1-2*poisson))
lamemu = Young2d / (3*(1 + poisson ))
visco = (4*1.6e-20 * joule / micron**3 * minute)  # 4*1 Pa.s --> includes the oseen factor

l0 = R
f0 = R #(2*kT*l0**2/Young2d)**(1/4)
u0 = f0**2/R
T0 = visco*(l0**3/f0**2) /Young2d

timestep = 0.01
simdt = 600 * timestep
#stop_sim_time = 200 * minute

stop_sim_time = 3000 # scaled sim time units

#timestep = 1e-3
#simdt = 200*timestep
#stop_sim_time = 2*T0

resolution = (Nphi, Ntheta)
chunk = 600


N_stat = 1

ratios = [0.001, 0.01, 0.03,0.08, 0.1, 0.3, 1 ]
ratios = [0.001, 0.003, 0.01, 0.03, 0.08]
ratios = [0.03, 0.01, 0.03, 0.04, 0.05, 0.06] 
ratios = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]

ratios = [0.005, 0.01, 0.05, 0.1, 0.5]
ratios = [0.05, 0.1, 0.5, 1, 5]

kappa = 100 * kT

alpha = [0., 0.003, 0.01, 0.03, 0.1]
alpha = [0., 0.2, 0.4, 0.6, 0.7]

#ratios = [1e-5, 1e-4, 1e-3]
#ratios = [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
#ratios = [0.01]
#kappas = N_stat* [kT/r for r in ratios]
#kappas.sort()

alphas = N_stat * alpha
alphas.sort()

# assign work
abs_idx = np.arange(my_task_id-1, len(alphas), num_tasks)
my_parameters = alphas[my_task_id-1:len(alphas):num_tasks]

rs = RandomState(MT19937(SeedSequence(12345678)))
for idx, parameter in enumerate(my_parameters):
    
    Young2d = FvK*kappa / R**2
    pressure = parameter*4*np.sqrt(kappa * Young2d) / R**2

    params = {'kappa':kappa, 'young':Young2d, 'kT':kT, 'p':pressure, 'R':R, 'Rc':1*R, 'poisson':0.3, 'visco':visco,
             'T0':100*minute, 'rampT':20*minute}
     
    output_filename = './runX/d3sh2f_varP_Rc1_fvk3e4_{0}_runX_R25.mat'.format(abs_idx[idx])
    mid_filename = 'snapshotsX_{0}'.format(abs_idx[idx])
    foldername = tmpdir


    d3shellsolver(params, output_filename, foldername, mid_filename, resolution,
                  chunk, timestep, simdt, stop_sim_time, seed=np.random.randint(100))
print('**********************************')
print('******** TASK FINISHED ***********')
print('**********************************')

