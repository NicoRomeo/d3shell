# d3shell

Dedalus script simulating a thin elastic shell in a viscous heat bath as a Langevin process. It can be ran serially or in parallel, and uses the built-in analysis framework to save data snapshots to HDF5 files. A conversion to .mat allows the output to be analyzed in a different spherical harmonic
spectral analysis framework.

The package requires dedalus3 and pyshtools to run. The execution scripts shell_multiple.py and shell_varp.py are designed to be run on MIT supercloud's SLURM scheduler jobarray.

This code has been developed for the paper Dynamics, scaling behavior, and control of nuclear wrinkling by Jonathan A. Jackson, Nicolas Romeo, Alexander Mietke, Keaton J. Burns, Jan F. Totz, Adam C. Martin, Jörn Dunkel, Jasmin Imran Alsous.

If you find it useful for your research, please cite the above paper.
