"""
Converts dedalus output to format compatible with MATLAB analysis code.
"""
from random import sample
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dedalus.public as d3
import pyshtools as sht
import scipy.io
import os
import re


def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def reshape_modes(data, ell_lay, m_lay, l_max):
    amps = np.zeros((l_max+1)**2)
    s2 = np.sqrt(2)
    print("data shape ", data.shape)
    print("spectral layount shape", ell_lay.shape)
    for l in range(l_max+1):
        ll = l*(l+1)
        amps_l = data[ell_lay == l]
        for m in range(-l, l+1):
            if m <0:
                amps[ll+m] =  1/s2 * amps_l[m_lay == m][1] + (-1)**(m)/s2 * amps_l[m_lay == -m][1]
            else:
                if m == 0:
                    amps[ll+m] = np.sum(amps_l[m_lay == 0])
                else:
                    amps[ll+m] = (-1)**m/s2 * amps_l[m_lay == m][0] + 1/s2 * amps_l[m_lay == -m][0]
    return amps

def shtransform(data, grid, l_max):
    cilm = sht.expand.SHExpandDH(data.T, norm=4, sampling=2, lmax_calc=l_max) # norm=4 is orthonormal SH
    #cilm = sht.expand.SHExpandDH(data.T, norm=4, sampling=1, lmax_calc=l_max) # norm=4 is orthonormal SH
    # turn the cilm (2, l_max +1, l_max +1) shaepd array into a (l_max+1)^2 array
    res = np.zeros(((l_max+1)**2,))
    for l in range(l_max+1):
        ll = l*(l+1)
        for m in range(-l,0):
            res[ll+m] = cilm[1, l, -m]
        for m in range(0, l+1):
            res[ll+m] = cilm[0, l, m]
        # is the m=0 case treated correctly?
    return res


def load_and_crop(infiles, l_max, dt, R, chunk, dealias):
    # Plot settings
    Nmodes = (l_max+1)**2
    
    
    # find all files 

    directory = os.path.dirname(infiles)
    basename = os.path.basename(infiles)
    len_basename = len(basename)

    size_chunk = chunk
    count = 0
    dtype = np.float64
    list_of_files = [files for files in os.listdir(directory) if files[:len_basename] == basename and files[-3:] == '.h5']
    sort_nicely(list_of_files)
    Ntimes = size_chunk * len(list_of_files)
    sample_times = dt * np.arange(Ntimes)
    modes_amplitude = np.zeros((Ntimes, Nmodes))
    u_norm = np.zeros((Ntimes, 1))
    for filename in list_of_files:
        with h5py.File(os.path.join(directory,filename), mode='r') as file:

            task = 'inplane'
            dset_u = file['tasks'][task]

            task = 'radial'
            dset = file['tasks'][task]
            phi = dset.dims[1][0][:].ravel()

            Nphi = len(phi)
            theta = dset.dims[2][0][:].ravel()

            Ntheta = len(theta)
            lat  = np.pi/2 - theta
            lon = phi - np.pi

            grid = np.meshgrid(lat, lon, indexing='ij')
            coords = d3.S2Coordinates('phi', 'theta')
            dist = d3.Distributor(coords, dtype=dtype)
            basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)
            m, ell = dist.coeff_layout.local_group_arrays(basis.domain, scales=1)
            len_cur = dset.shape[0]
            for index in range(0, len_cur):
                data_slices = (index, slice(None), slice(None))
                data = dset[data_slices]
                modes_amplitude[index+count,:] =  shtransform(data, grid, l_max)
                u_norm[index+count] = dset_u[data_slices][:]
            count += len_cur
    return sample_times, modes_amplitude, u_norm

def save_to_matlab(infiles, outfile, l_max, simdt, R, chunk, dealias, params):
    sample_times, modes, u_norm  = load_and_crop(infiles, l_max, simdt, R, chunk, dealias)
    scipy.io.savemat(outfile, {'times':sample_times.T, 'f':modes, "radius":R, 'u_norm':u_norm,
    "kappa":params['kappa'], "pressure":params['p'], "young":params['young'], 'kT':params['kT']})
    
def load_and_crop_eff(infiles, l_max, dt, R, chunk, dealias):
    # Plot settings
    Nmodes = (l_max+1)**2
    
    
    # find all files 

    directory = os.path.dirname(infiles)
    basename = os.path.basename(infiles)
    len_basename = len(basename)

    size_chunk = chunk
    count = 0
    dtype = np.float64
    list_of_files = [files for files in os.listdir(directory) if files[:len_basename] == basename and files[-3:] == '.h5']
    sort_nicely(list_of_files)
    Ntimes = size_chunk * len(list_of_files)
    sample_times = dt * np.arange(Ntimes)
    modes_amplitude = np.zeros((Ntimes, Nmodes))
    for filename in list_of_files:
        with h5py.File(os.path.join(directory,filename), mode='r') as file:

            task = 'radial'
            dset = file['tasks'][task]
            phi = dset.dims[1][0][:].ravel()

            Nphi = len(phi)
            theta = dset.dims[2][0][:].ravel()

            Ntheta = len(theta)
            lat  = np.pi/2 - theta
            lon = phi - np.pi

            grid = np.meshgrid(lat, lon, indexing='ij')
            coords = d3.S2Coordinates('phi', 'theta')
            dist = d3.Distributor(coords, dtype=dtype)
            basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)
            m, ell = dist.coeff_layout.local_group_arrays(basis.domain, scales=1)
            len_cur = dset.shape[0]
            for index in range(0, len_cur):
                data_slices = (index, slice(None), slice(None))
                data = dset[data_slices]
                modes_amplitude[index+count,:] =  shtransform(data, grid, l_max)
            count += len_cur
    return sample_times, modes_amplitude

def save_to_matlab_eff(infiles, outfile, l_max, simdt, R, chunk, dealias, params):
    sample_times, modes  = load_and_crop(infiles, l_max, simdt, R, chunk, dealias)
    scipy.io.savemat(outfile, {'times':sample_times.T, 'f':modes, "radius":R,
    "kappa":params['kappa'], "pressure":params['p'], "young":params['young'], 'kT':params['kT']})

if __name__ == '__main__':

    l_max = 25
    Nmodes = (l_max +1)**2

    import pathlib

    sample_times, modes  = load_and_crop("./snapshots/snapshots_s", l_max, 0.005, 1, 10, 2)
    
    print(modes.shape)
    output_file = "output_test.mat"
    scipy.io.savemat(output_file, {'times':sample_times.T, 'f':modes})
