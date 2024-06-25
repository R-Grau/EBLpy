# EBL_fit_MC
Repository to simulate EBL absorbed blazar spectrum observations with MAGIC, LST-1, or CTAO-North alpha configuration and make an EBL density profile likelihood with it.
Also to simulate different Poisson realizations of the same observation and plot in a 2D histogram of the EBL density profile likelihood.
Finally, scanEBL does the profile likelihood with real data from MAGIC or LST-1. 

# Simulation
## MAGIC
### Requirements: 

-Output_flute.root of real data to get the background and the IRF.

-MARS and root installed

-Uproot installed

### Instructions:

1-Use the file in the MAGIC folder to generate the IRF

2-Generate the config files with EBL_MC_config_creator.ipynb (you need one general config file and one file for each simulated spectra function and fit function)

3-Run EBL_MC.py with "python EBL_MC.py number" where number is the iteration number that will be used as a seed for the random number generator of the Poisson realization.

3*- You can run EBL_MC.py in a cluster in parallel with different iteration numbers to speed up the process

## LST-1 and CTAO-N
### Requirements:

-DL3 file with real data to get the background and IRF (LST-1) or CTAO-N IRFs from (https://www.ctao.org/for-scientists/performance/ (CTAO-N)

-gammapy installed

### Instructions:

1-Use the file in LST-1 or CTAO-N folder to get the IRFs

2-Generate the config files with EBL_MC_config_creator.ipynb (you need one general config file and one file for each simulated spectra function and fit function)

3-Run EBL_MC.py with "python EBL_MC.py number" where number is the iteration number that will be used as a seed for the random number generator of the Poisson realization.

3*- You can run EBL_MC.py in a cluster in parallel with different iteration numbers to speed up the process

# Real data
### Requirements: 

Same as for the simulation, depending on the telescope
### Instructions:

1-Use the file in LST-1 or MAGIC folder to get the IRFs

2-Change the second cell parameters to match your needs (Telescope, redshift of the source, Telescope used,...)

3-Run all the other cells.
