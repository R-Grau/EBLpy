#Before running this script you need to generate all the needed configuration files with the script "EBL_MC_final_config_creator.ipynb"
#You need the general config, one config_data for every simulated spectra and one config_fit for every fit function

import numpy as np
import scipy as sc
from iminuit import Minuit
import pandas as pd
import scipy.interpolate as interpolate
from scipy.integrate import quad
from matplotlib import colors
from EBL_MC_final_functions import *
from scipy.stats import norm
from scipy.stats import poisson
from joblib import Parallel, delayed
import yaml
import matplotlib.pyplot as plt
import h5py
import time
import sys
import os
import uproot
from ebltable.tau_from_model import OptDepth

systematics = 0.07 #Gaussian systematic errors to be added to the simulated data
Syst = True #add systematics to the analysis
Eshift = False #whether we want to add a shift on the Energy
migmatshift = 0.15 #how much we want to shift the migration matrix (with gaussian probability) (useless when Eshift = False)
randomseed = int(sys.argv[1]) #random seed for the simulation added as an argument when running the script

Emin = 0.06  #minimum energy of the simulated data in TeV
Emax = 20. #maximum energy of the simulated data in TeV
scan_method = 5 # 0 for normal scanning, 1 for scanning and retryng failed fits, 2 for scanning and scanning again with another initial guess, 3 for scanning, retrying failed fits and 
                #scanning again with another initial guess and 4 for doing every alpha by itself (no chained initial guess)
initial_guess_pos = 2.05 #Position of the initial guess before the scan              
other_initial_guess_position = -0.05 #this is only used scan_method = 2, 3 and 5     
migmatmaxu = 0.51 #maximum value of the migration matrix uncertainty relative to the value (to discard points with few MC statistics)

Extratxt = "FINAL_with_syst_analysis_dominguez" #text to add to the name of the output files
pathstring = "/data/magic/users-ifae/rgrau/EBL-splines/"#path where the data files are and where the output files will be saved

#Load the general configuration file

Telescope, niter, Energy_migration, Forward_folding, IRF_u, Background, fit_n, Spectrum_fn = general_config()

#Now first we create the data for every intrinsic spectrum function and for every intrinsic spectrum we will fit the different fit functions


for Spectrum_func_name in Spectrum_fn: #loop over the different intrinsic spectrum functions
    #first load the configuration file for the intrinsic spectrum function
    Source_flux, Observation_time, Background_scale, Norm, Ph_index, LP_curvature, E_cut, d, Source_z, EBL_Model_sim = config_data(Spectrum_func_name)
    
    if Spectrum_func_name == "PWL": #define the intrinsic spectrum if the function is a Power-Law
        def dNdE_absorbed(K, E, Norm, Ph_index, tau):
            dNdE = K / ((E/Norm)**Ph_index)
            m_tau = -tau
            return dNdE * np.exp(m_tau)

    elif Spectrum_func_name == "LP": #define the intrinsic spectrum if the function is a Log-Parabola
        def dNdE_absorbed(K, E, Norm, Ph_index, b, tau):
            dNdE = K * np.power((E/Norm), (-Ph_index - b * b * np.log10(E/Norm)))
            m_tau = -tau
            return dNdE * np.exp(m_tau)
        
    elif Spectrum_func_name == "EPWL": #define the intrinsic spectrum if the function is an Exponential cut-off Power-Law
        def dNdE_absorbed(K, E, Norm, Ph_index, E_cut, tau):
            dNdE = K / ((E/Norm)**Ph_index) * np.exp(-E/(E_cut*E_cut))
            m_tau = -tau
            return dNdE * np.exp(m_tau)

    elif Spectrum_func_name == "ELP": #define the intrinsic spectrum if the function is an Exponential cut-off Log-Parabola
        def dNdE_absorbed(K, E, Norm, Ph_index, b, E_cut, tau):
            dNdE = K * np.power((E/Norm), (-Ph_index - b * b * np.log10(E/Norm))) * np.exp(-E / (E_cut * E_cut))
            m_tau = -tau
            return dNdE * np.exp(m_tau)

    elif Spectrum_func_name == "SEPWL": #define the intrinsic spectrum if the function is a Super-Exponential cut-off Power-Law
        def dNdE_absorbed(K, E, Norm, Ph_index, E_cut, d, tau):
            dNdE = K / ((E/Norm)**Ph_index) * np.exp(-(E/(E_cut*E_cut)**d))
            m_tau = -tau
            return dNdE * np.exp(m_tau)
    else:
        raise Exception('The simulated spectrum "{func}" has not been implemented.'.format(func = Spectrum_func_name))
        
 #########################Maybie this can be moved before or later
    if Forward_folding:
        if Telescope == "CTAN_alpha":
            print("CTAN-alpha to be configured soon")
                
        elif Telescope == "MAGIC":
            Noffregions = 3
            def m2LogL(params):
                xdata = Etrue
                mtau_fit = -tau_fit
                if IRF_u:
                    mu_gam, mu_gam_u = dNdE_to_mu_MAGIC_IRF((fit_func(xdata, params) * np.exp(mtau_fit * alpha)), Ebinsw_Etrue, migmatval, migmaterr, Eest)
                    if Syst:
                        mu_gam_u = np.sqrt(mu_gam_u**2 + (mu_gam * systematics)**2)
                    mu_gam_final_u = mu_gam_u[minbin:maxbin]

                else:
                    mu_gam = dNdE_to_mu_MAGIC((fit_func(xdata, params) * np.exp(mtau_fit * alpha)), Ebinsw_Etrue, migmatval, Eest)

                mu_gam_final = mu_gam[minbin:maxbin]
                Non_final = Non[minbin:maxbin] 
                Noff_final = Noff[minbin:maxbin]
                min_num_gauss = 20

                res = np.ones(len(Non_final)) * 999999999
                for i in range(len(Non_final)):
                    if IRF_u:
                        if ((Non_final[i] >= min_num_gauss) & (Noff_final[i] >= min_num_gauss)):
                            res[i] = Gauss_logL_IRF(Non_final[i], Noff_final[i], mu_gam_final[i], mu_gam_final_u[i], Noffregions)
                        elif Non_final[i] == 0:
                            res[i] = Poisson_logL_Non0_IRF(Non_final[i], Noff_final[i], mu_gam_final[i], mu_gam_final_u[i], Noffregions)
                        elif Noff_final[i] == 0:
                            res[i] = Poisson_logL_Noff0_IRF(Non_final[i], Noff_final[i], mu_gam_final[i], mu_gam_final_u[i], Noffregions)
                        elif mu_gam_final[i] < 1e-6:
                            res[i] = Poisson_logL_small_mugam_IRF(Non_final[i], Noff_final[i], mu_gam_final[i], mu_gam_final_u[i], Noffregions)
                        elif mu_gam_final_u[i] == 0:
                            res[i] = Poisson_logL_noIRF_IRF(Non_final[i], Noff_final[i], mu_gam_final[i], mu_gam_final_u[i], Noffregions)
                        elif (Non_final[i] != 0) & (Noff_final[i] != 0):
                            res[i] = Poisson_logL_else_IRF(Non_final[i], Noff_final[i], mu_gam_final[i], mu_gam_final_u[i], Noffregions)
                    
                    else:
                        if ((Non_final[i] >= min_num_gauss) & (Noff_final[i] >= min_num_gauss)):
                            res[i] = Gauss_logL(Non_final[i], Noff_final[i], mu_gam_final[i], Noffregions)
                        elif Non_final[i] == 0:
                            res[i] = Poisson_logL_Non0(Non_final[i], Noff_final[i], mu_gam_final[i], Noffregions)
                        elif Noff_final[i] == 0:
                            res[i] = Poisson_logL_Noff0(Non_final[i], Noff_final[i], mu_gam_final[i], Noffregions)
                        elif (Non_final[i] != 0) & (Noff_final[i] != 0):
                            res[i] = Poisson_logL_else(Non_final[i], Noff_final[i], mu_gam_final[i], Noffregions) 

                return np.sum(res)

        def fit(initial_guess):
            
            m2LogL.errordef = Minuit.LIKELIHOOD
            m = Minuit(m2LogL, initial_guess)
            if fit_func_name == "MBPWL": #defines limits to faster and better find the minimum. Can be changed if the convergence fails.
                MBPWL_limits = ([(1e-6, 1e-3), (-4., 5.)])
                errors = ([1e-7, 0.01])
                for i in range(knots):
                    MBPWL_limits.append((0., 5.))
                    errors.append(0.01)
                m.limits = MBPWL_limits
            elif fit_func_name == "PWL":
                m.limits = ([(1e-7, 1e-3), (None, 6.)])
                errors = [1e-7, 0.01]
            elif fit_func_name == "LP" or fit_func_name == "freeLP":
                m.limits = ([(1e-7, 1e-3), (None, 6.), (None, None)])
                errors = [1e-7, 0.01, 0.1]
            elif fit_func_name == "EPWL":
                m.limits = ([(1e-7, 1e-3), (None, None), (None, None)])
                errors = [1e-8, 1.0, np.sqrt(500.)]
            elif fit_func_name == "ELP":
                m.limits = ([(1e-7, 1e-3), (-2., None), (None, None), (None, None)])
                errors = [1e-8, 1., 0.1, np.sqrt(500.)]
            elif fit_func_name == "SEPWL":
                m.limits = ([(1e-7, 1e-3), (None, None), (None, None), (None, None)])
                errors = [1e-8, 1.0, np.sqrt(500.), 1.]
            elif fit_func_name == "SELP":
                m.limits = ([(1e-7, 1e-3), (-2., None), (None, None), (None, None), (None, None)])
                errors = [1e-8, 1., 0.1, np.sqrt(500.), 0.1]
            #m.tol = 1e-6
            #m.strategy = 2
            m.errors = errors
            
            m.migrad()
            return m
    
 ################################################################
    
    if Telescope == "CTAN_alpha": 
        print("Not implemented yet")

    elif Telescope == "MAGIC": #Simulate the MAGIC data
        
        Bckg = uproot.open("{0}Output_flute.root:hEstBckgE".format(pathstring))#load background values
        bckgmu_final = Bckg.values() #counts in 42480s (can be normalized for any time but as the migmatrix is for that time, only use that time).

        migrmatrix = uproot.open("{0}fold_migmatrix.root:mig_matrix".format(pathstring)) #load migration matrix
        migmatval = migrmatrix.values() #m^2 * s #values
        if IRF_u:
            migmaterr = migrmatrix.errors()
            migmatval[(migmaterr/migmatval)>migmatmaxu] = 0
            migmaterr[(migmaterr/migmatval)>migmatmaxu] = 0
        migmatxEtrue = migrmatrix.axis("x").edges()/1e3 #TeV #edge values of X axis of the migration matrix (True Energy)
        migmatyEest = migrmatrix.axis("y").edges()/1e3 #TeV #edge values of Y axis of the migration matrix (Estimated Energy)

        Eest = migrmatrix.axis("y").centers()/1e3 #TeV #center values of X axis of the migration matrix (True Energy)
        Etrue = migrmatrix.axis("x").centers()/1e3 #TeV #center values of Y axis of the migration matrix (Estimated Energy)
        E_final = Etrue
        Usedbins = np.where((Emin <= Eest) & (Eest <= Emax))
        minbin = Usedbins[0][0]
        maxbin = Usedbins[0][-1] + 1
        Eest_final = Eest[minbin:maxbin]
        
        #tau_sim = tau_interp(Etrue, Source_z, EBL_Model_sim, kind_of_interp = "log") #old, before adding ebltable package #interpolate the tau values to have the same bins as the migration matrix and the data.
        tau1 =  OptDepth.readmodel(model=EBL_Model_sim)
        tau_sim = tau1.opt_depth(Source_z, Etrue) #interpolate the tau values to have the same bins as the migration matrix and the data.


        Ebinsw_final = migmatyEest[1:] - migmatyEest[:-1] #compute the bin width of the final energy bins
        Ebinsw_Etrue = migmatxEtrue[1:] - migmatxEtrue[:-1] #compute the bin width of Etrue energy bins

        if Spectrum_func_name == "PWL":
            dNdEa = dNdE_absorbed(Source_flux, Etrue, Norm, Ph_index, tau_sim) #use the previously defined dNdE function 

        elif Spectrum_func_name == "LP":
            dNdEa = dNdE_absorbed(Source_flux, Etrue, Norm, Ph_index, LP_curvature, tau_sim) #use the previously defined dNdE function 

        elif Spectrum_func_name == "EPWL":
            dNdEa = dNdE_absorbed(Source_flux, Etrue, Norm, Ph_index, E_cut, tau_sim) #use the previously defined dNdE function

        elif Spectrum_func_name == "ELP":
            dNdEa = dNdE_absorbed(Source_flux, Etrue, Norm, Ph_index, LP_curvature, E_cut, tau_sim) #use the previously defined dNdE function

        elif Spectrum_func_name == "SEPWL":
            dNdEa = dNdE_absorbed(Source_flux, Etrue, Norm, Ph_index, E_cut, d, tau_sim) #use the previously defined dNdE function


        if Eshift:
            mu_vec_final = dNdE_to_mu_MAGIC_Eshift(dNdEa, Ebinsw_Etrue, migmatval, Eest, migmatshift, randomseed, Etrue) #get the dNdE to the needed mu values for the likelihood.
        else:
            mu_vec_final = dNdE_to_mu_MAGIC(dNdEa, Ebinsw_Etrue, migmatval, Eest) #get the dNdE to the needed mu values for the likelihood.

    else:
        raise Exception('The telescope "{func}" has not been implemented.'.format(func = Telescope))


    xdata = E_final
    datetime = time.strftime("%Y%m%d%H%M")
    iter = int(sys.argv[1])
        
    for fit_func_name in fit_n: #loop over the different fit functions
        print("Starting function {func} for iter {iter}".format(func = fit_func_name, iter = iter))
        EBL_Model_fit, initial_guess_0, step, true_alpha_max, true_alpha_min, knots, Efirst, DeltaE, Source_z = config_fit(fit_func_name)
        #tau_fit = tau_interp(Etrue, Source_z, EBL_Model_fit, kind_of_interp = "log")
        tau2 =  OptDepth.readmodel(model=EBL_Model_fit)
        tau_fit = tau2.opt_depth(Source_z, Etrue)

        first_bin = true_alpha_min - step 
        last_bin = true_alpha_max + step


        fit_func = fit_func_select(fit_func_name, knots, Efirst, DeltaE) #define the fit function for the minimization
        # name the folder where the data will be stored and the datafile name
        if Spectrum_func_name == "LP":
            if not os.path.exists('{path}EBL{niter}_{func1}{curv}_{func2}_{telescope}_with_{systematics}_Systematics_{extra}'.format(path = pathstring, curv = LP_curvature, func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt)):
                os.mkdir('{path}EBL{niter}_{func1}{curv}_{func2}_{telescope}_with_{systematics}_Systematics_{extra}'.format(path = pathstring, curv = LP_curvature, func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt))
                os.popen('cp {path}EBL_MC_config_data_{func1}.yml {path}EBL{niter}_{func1}{curv}_{func2}_{telescope}_with_{systematics}_Systematics_{extra}/EBL_MC_config_data_{func1}.yml'.format(path = pathstring, curv = LP_curvature, func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt))                
                os.popen('cp {path}EBL_MC_config_fit_{func2}.yml {path}EBL{niter}_{func1}{curv}_{func2}_{telescope}_with_{systematics}_Systematics_{extra}/EBL_MC_config_fit_{func2}.yml'.format(path = pathstring, curv = LP_curvature, func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt))
            hdf5filename = "{path}EBL{niter}_{func1}{curv}_{func2}_{telescope}_with_{systematics}_Systematics_{extra}/EBL_mult_nit{nit}of{niter}_{datetime}.hdf5".format(path = pathstring, curv = LP_curvature, func1 = Spectrum_func_name, func2 = fit_func_name ,nit = iter, niter = niter, datetime = datetime, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt)
            savefile = h5py.File(hdf5filename, "w")
        else:
            if not os.path.exists('{path}EBL{niter}_{func1}_{func2}_{telescope}_with_{systematics}_Systematics_{extra}'.format(path = pathstring, func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt)):
                os.mkdir('{path}EBL{niter}_{func1}_{func2}_{telescope}_with_{systematics}_Systematics_{extra}'.format(path = pathstring, func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt))
                os.popen('cp {path}EBL_MC_config_data_{func1}.yml {path}EBL{niter}_{func1}{curv}_{func2}_{telescope}_with_{systematics}_Systematics_{extra}/EBL_MC_config_data_{func1}.yml'.format(path = pathstring, curv = LP_curvature, func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt))                                
                os.popen('cp {path}EBL_MC_config_fit_{func2}.yml {path}EBL{niter}_{func1}_{func2}_{telescope}_with_{systematics}_Systematics_{extra}/EBL_MC_config_fit_{func2}.yml'.format(path = pathstring, func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt))
            hdf5filename = "{path}EBL{niter}_{func1}_{func2}_{telescope}_with_{systematics}_Systematics_{extra}/EBL_mult_nit{nit}of{niter}_{datetime}.hdf5".format(path = pathstring, func1 = Spectrum_func_name, func2 = fit_func_name ,nit = iter, niter = niter, datetime = datetime, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt)
            savefile = h5py.File(hdf5filename, "w")

        #do the forward folding:
        if Forward_folding:
            if Background != True:
                print("The forward folding is done with background")

            def process2(iter_num, alphas, mu_on, mu_off):
                chisqs = np.ones(len(alphas)) * 99999999
                global alpha, Non, Noff
                alpha = initial_guess_pos
                rng_num = iter_num
                my_generator = np.random.default_rng(rng_num)
                Non, Noff = my_generator.poisson(mu_on), my_generator.poisson(Noffregions * mu_off)
                Non_u, Noff_u = np.sqrt(Non), np.sqrt(Noff)

                if scan_method == 4:
                    for i, alpha0 in enumerate(alphas):
                        alpha = alpha0
                        things = fit(initial_guess = initial_guess_0)
                        if things.valid == False:
                            print("Function {0} did not minimize properly the {1} intrinsic spectra for iteration {2}".format(fit_func_name, Spectrum_func_name, iter))
                        else:
                            chi2 = m2LogL(things.values)
                            chisqs[i] = chi2
                
                else:
                    things = fit(initial_guess=initial_guess_0)
                    initial_guess_mat = ig_mat_create(fit_func_name, alphas, knots)
                    initial_guess_mat[0] = things.values
                    for i, alpha0 in enumerate(alphas):
                        alpha = alpha0
                        initial_guess = initial_guess_mat[i]
                        if alpha == initial_guess_pos:
                            initial_guess = initial_guess_mat[0]
                        things = fit(initial_guess = initial_guess)
                        if things.valid == False:
                            print("Function {0} did not minimize properly the {1} intrinsic spectra for iteration {2} for alpha = {3}".format(fit_func_name, Spectrum_func_name, iter, alpha))
                            #print("Function {0} minimized properly the {1} intrinsic spectra for iteration {2}".format(fit_func_name, Spectrum_func_name, iter))
                        else:    
                            chi2 = m2LogL(things.values)
                            chisqs[i] = chi2

                        if i < len(alphas):
                            initial_guess_mat[i+1] = things.values
                    if scan_method == 1 or scan_method == 3:
                        for i, alpha0 in enumerate(reversed(alphas)):
                            j = len(alphas) - i - 1
                            if chisqs[j] == 99999999 or np.isnan(chisqs[j]):
                                print("Retrying alpha = {0} with different initial guess".format(alpha0))
                                alpha = alpha0
                                if i == 0:
                                    initial_guess = initial_guess_mat[0]
                                else:
                                    initial_guess = initial_guess_mat[j+2]
                                things = fit(initial_guess = initial_guess)
                                if things.valid == False:
                                    print("Function {0} did not minimize properly the {1} intrinsic spectra for iteration {2} with the other initial guess".format(fit_func_name, Spectrum_func_name, iter))
                                    print("The initial guess was: ", initial_guess)
                                    break
                                else:    
                                    print("Function {0} minimized properly the {1} intrinsic spectra for iteration {2} in 2nd try with the other initial guess".format(fit_func_name, Spectrum_func_name, iter))
                                    chi2 = m2LogL(things.values)
                                    # print("chi2: ", chi2)
                                    chisqs[j] = chi2
                    if scan_method == 5:
                        for i, alpha0 in enumerate(reversed(alphas)):
                            j = len(alphas) - i - 1
                            alpha = alpha0
                            if i == 0:
                                initial_guess = initial_guess_mat[0]
                            else:
                                initial_guess = initial_guess_mat[j+2]
                            things = fit(initial_guess = initial_guess)
                            if things.valid == False:
                                print("Function {0} did not minimize properly the {1} intrinsic spectra for iteration {2} with the other initial guess".format(fit_func_name, Spectrum_func_name, iter))
                            else:    
                                print("Function {0} minimized properly the {1} intrinsic spectra for iteration {2} in 2nd try with the other initial guess".format(fit_func_name, Spectrum_func_name, iter))
                                chi2 = m2LogL(things.values)
                                if (chi2 < chisqs[j] and not(np.isnan(chi2))) or np.isnan(chisqs[j]):
                                    chisqs[j] = chi2

                print("Finished minimization for function {0} of the {1} intrinsic spectra for iteration {2}".format(fit_func_name, Spectrum_func_name, iter))

                return chisqs
            
            alphas = alphas_creation(initial_guess_pos, first_bin, last_bin, step) #TODO change first bin and last bint in config file.

            my_generator2 = np.random.default_rng(iter)
            if systematics == 0.0:
                mu_on = mu_vec_final + bckgmu_final
            else:
                mu_on = my_generator2.normal(mu_vec_final, systematics * mu_vec_final) + bckgmu_final #to add some systematics to try to fit the real results
            mu_off = bckgmu_final

            chisqs1 = process2(iter, alphas, mu_on, mu_off)
            #if the first try did not work, we try to minimize it with another initial guess position
            if scan_method == 2 or scan_method == 3:
                if (chisqs1 == 99999999).any() or (chisqs1 == np.inf).any() or (np.isnan(chisqs1)).any(): #maybie we don't need this any more
                    print("As function {0} did not minimize properly the {1} intrinsic spectra for iteration {2} in the 2nd try with the other initial guess, we will try to minimize it with another initial guess position".format(fit_func_name, Spectrum_func_name, iter))
                    alphas2 = alphas_creation(other_initial_guess_position, first_bin, last_bin, step)
                    chisqs2 = process2(iter, alphas2, mu_on, mu_off)

                    #need to reorder the vectors in order to have the same order as the one with the previous initial guess.
                    #FIXME make all to go from 0 to max
                    initial_pos = np.where(alphas == alphas2[0])[0][0]
                    indices = np.argsort(alphas)
                    index_init = np.where(indices == initial_pos)[0][0]
                    chisqs2 = np.concatenate((chisqs2[indices[index_init:]], np.flip(chisqs2[indices[:index_init]])))
                    chisqs = np.minimum(chisqs1, chisqs2)
                    nan_indices = np.isnan(chisqs1) | np.isnan(chisqs2)
                    chisqs[nan_indices] = np.where(np.isnan(chisqs1), chisqs2, chisqs1)[nan_indices]
                else:
                    chisqs = chisqs1

            elif scan_method == 5:
                alphas2 = alphas_creation(other_initial_guess_position, first_bin, last_bin, step)
                chisqs2 = process2(iter, alphas2, mu_on, mu_off)

                #need to reorder the vectors in order to have the same order as the one with the previous initial guess.
                #FIXME make all to go from 0 to max
                initial_pos = np.where(alphas == alphas2[0])[0][0]
                indices = np.argsort(alphas)
                index_init = np.where(indices == initial_pos)[0][0]
                chisqs2 = np.concatenate((chisqs2[indices[index_init:]], np.flip(chisqs2[indices[:index_init]])))
                chisqs = np.minimum(chisqs1, chisqs2)#make that if there is a nan use the other one
                nan_indices = np.isnan(chisqs1) | np.isnan(chisqs2)
                chisqs[nan_indices] = np.where(np.isnan(chisqs1), chisqs2, chisqs1)[nan_indices]

            else:
                chisqs = chisqs1

            alphas3 = alphas[np.where((alphas <= true_alpha_max) & (alphas >= true_alpha_min))]
            chisqs3 = chisqs[np.where((alphas <= true_alpha_max) & (alphas >= true_alpha_min))]
            order = np.argsort(alphas3)
            alphas_save = np.take_along_axis(alphas3, order, axis=-1)
            chisqs_save = np.take_along_axis(chisqs3, order, axis=-1)
            dset = savefile.create_dataset("alphas", data = alphas_save, dtype='float')
            dset = savefile.create_dataset("chisqs", data = chisqs_save, dtype='float')
            savefile.close()

        else:
            print("Use Forward Folding please")