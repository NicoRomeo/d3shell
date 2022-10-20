"""
Dedalus script simulating a thin elastic shell in a viscous heat bath as a Langevin process. It can be
ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. A conversion to .mat allows the output to be analyzed in a different spherical harmonic
spectral analysis framework


To run and plot using e.g. 4 processes, use the associated scripts shell_multiple.py and shell_varP.py with e.g.
    $ mpiexec -n 4 python3 shell_multiple.py

Inspired from Spherical shallow water (sphere IVP) example from dedalus 
"""

from time import time
import numpy as np
import dedalus.public as d3
from dedalus.core.operators import SphereEllProduct
from convert2matlab import save_to_matlab
from convert2matlab import save_to_matlab_eff
import logging
logger = logging.getLogger(__name__)


def d3shellsolver(params, output_filename, foldername, mid_filename, resolution, chunk, timestep, simdt, stop_sim_time, seed=None):
    """ A dedalus-based SDE solver for the stochastic elastic sphere 
    Inputs:
        params - dictionary with keys kappa, young, kT, p, R
        output_filename - string (ends in .mat)
        resolution - tuple of integers (2N, N)
        chunk - integer number of timesteps at which to record solution
        timestep - in simulation units
        simdt - sample time for record ( integer * timestep )
        stop_sim_time - final time of simulation (in simulation units)
    Output to file output_filename as a mat file using the convert2matlab function. """
    logger = logging.getLogger(__name__)

    if seed is not None:
        np.random.seed(seed=seed)
        
    Nphi, Ntheta = resolution

    R = params['R']
    Rc = params['Rc']
    kT = params['kT']
    kappa = params['kappa']
    Young2d = params['young']
    poisson = params['poisson']
    pressure = params['p']
    
    visco = params['visco']

    lameL = Young2d * poisson / ((1 - poisson )*(1-2*poisson))
    lamemu = Young2d / (3*(1 + poisson ))
    pc = 4*np.sqrt(kappa*Young2d)/R**2
    
    logger.info('============================')
    logger.info('============================')
    logger.info('kappa/kT = {0}, R = {1}, p/pc = {2}'.format(kappa/kT, R, pressure/pc))
    #Rc = 10*R

    std_noise = np.sqrt(2*kT / (np.pi* R*R * timestep))
    dtype = np.float64

    dealias = 4/2
    # Bases
    coords = d3.S2Coordinates('phi', 'theta')
    dist = d3.Distributor(coords, dtype=dtype)
    basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

    m, ell = dist.coeff_layout.local_group_arrays(basis.domain, scales=1)

    # Fields
    u = dist.VectorField(coords, name='u', bases=basis)
    f = dist.Field(name='f', bases=basis) # f here is f/R from the notes

    f_noise = dist.Field(name='f_noise', bases=basis)
    u_noise = dist.VectorField(coords, name='u_noise', bases=basis)

    # Substitutions
    grad_u = d3.Gradient(u)
    t_grad_u = d3.TransposeComponents(grad_u)


    def oseen(l,r):
        out =  (l + (l==0))*(l+1)/ (visco/ R*(2*l+1)*(2*l*l + 2*l- 1 + 2*(l==0)))
        return out

    def ftd(l,r):
        out =  np.sqrt((l + (l==0))*(l+1)/ (visco/ R*(2*l+1)*(2*l*l + 2*l- 1 + 2*(l==0))))
        return out

    ell_prod = lambda A: SphereEllProduct(A, coords, oseen)
    noise_prod = lambda A: SphereEllProduct(A, coords, ftd)

    def fill_noise(timestep):
        f_noise.fill_random(layout='c', seed=None)
        u_noise.fill_random(layout='c', seed=None)
        f_noise['c'] *= std_noise / np.sqrt(1 + (m==0)) # extra normalization for m==0 modes
        u_noise['c'] *= std_noise / np.sqrt(1 + (m==0))
        

    # Initial conditions: perturbation
    phi, theta = dist.local_grids(basis)
    hpert = 1e-9
    f.fill_random('g')
    f.low_pass_filter(scales=1/2)
    f['g'] *= hpert 

    l0 = R # lengthscale - radius
    f0 = R # deformation scale - radius
    u0 = f0**2/R
    T0 = visco*(l0**3/f0**2) /Young2d #this matches timescale ot the characterisitc timescale of the cubic non-linear term in the PDE.
    
    # Problem
    p = pressure
    noise_fact = np.sqrt(T0)/f0
    noise_fact_u = np.sqrt(T0)/u0
    problem = d3.IVP([u, f], namespace=locals()) # this is f/R, oriented inwards and u/R
    u_rhs = " T0*ell_prod(lameL/2 * f0**2/u0 * grad(dot(grad(f), grad(f)))) + T0*f0**2/u0*lamemu * ell_prod(lap(f)*grad(f)+dot(grad(grad(f)),grad(f))) + noise_fact_u*noise_prod(u_noise)"
    problem.add_equation("dt(u) - T0*ell_prod(lamemu*lap(u)) - T0*(lameL+lamemu)*ell_prod(grad(div(u))) + 2*T0*(lameL+lamemu)/Rc * f0/u0 *ell_prod(grad(f)) = " + u_rhs)
    f_rhs = " ell_prod(p *T0/f0 -(lamemu+lameL)/Rc *T0*f0* (dot(grad(f), grad(f))+ 2* f*lap(f))) + ell_prod((lameL/2 + lamemu) * T0*f0**2 * div(dot(grad(f), grad(f)) * grad(f))) " \
    + "+ lameL * u0*T0*ell_prod(div(div(u) * grad(f)))  + lamemu * u0*T0*ell_prod(div(dot(grad_u+t_grad_u, grad(f))))+ noise_prod(f_noise)*noise_fact"
    problem.add_equation("dt(f) + ell_prod(kappa*T0*lap(lap(f))) + 4*(lameL+lamemu)/Rc**2*T0 * ell_prod(f) - 2*(lameL+lamemu)/Rc *T0*u0/f0*ell_prod(div(u)) = " + f_rhs)
    
    # Solver
    solver = problem.build_solver(d3.RK111)
    solver.stop_sim_time = stop_sim_time # in units of T0!

    # Analysis
    snapshots = solver.evaluator.add_file_handler(mid_filename, sim_dt=simdt, max_writes=chunk)
    snapshots.add_task(f0*f, name='radial') # in units of R
    snapshots.add_task(u0**2*d3.ave(d3.dot(u,u)), name='inplane') # in units of R^2

    flow = d3.GlobalFlowProperty(solver, cadence=chunk)
    flow.add_property(d3.ave(f**2), name='fluct')

    # Main loop
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            fill_noise(timestep)
            solver.step(timestep)
            if (solver.iteration-1) % chunk == 0:
                logger.info('Iteration=%i, fluct=%e Time=%e, dt=%e' %(solver.iteration, flow.max('fluct'), solver.sim_time, timestep))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()
        logger.info("simulation over - saving to MATLAB")
        l_max = 100
        save_to_matlab('./'+ mid_filename+ '/' + mid_filename+'_', output_filename, l_max, simdt, R, chunk, dealias, params)
        
        logger.info("MATLAB export complete.")
