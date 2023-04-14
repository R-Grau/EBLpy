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

systematics = 0.07
Emin = 0.06
Emax = 15.
Chain_guess = True
Extratxt = "FINAL_5lim"
pathstring = "/data/magic/users-ifae/rgrau/EBL-splines/"#"/home/rgrau/Desktop/EBL_pic_sync/"#"/data/magic/users-ifae/rgrau/EBL-splines/"

#Load the general configuration file

Telescope, niter, Energy_migration, Forward_folding, IRF_u, Background, fit_n, Spectrum_fn = general_config()

#Now first we create the data for every intrinsic spectrum function and for every intrinsic spectrum we will fit the different fit functions


for Spectrum_func_name in Spectrum_fn: #loop over the different intrinsic spectrum functions
    Source_flux, Observation_time, Background_scale, Norm, Ph_index, LP_curvature, Source_z, EBL_Model = config_data(Spectrum_func_name)
    
    if Spectrum_func_name == "PWL": #define the intrinsic spectrum if the function is a Power-Law
        def dNdE_absorbed(K, E, Norm, Ph_index, tau):
            dNdE = K / ((E/Norm)**Ph_index)
            m_tau = -tau
            return dNdE * np.exp(m_tau)

    elif Spectrum_func_name == "LP": #define the intrinsic spectrum if the function is a Log-Parabola
        def dNdE_absorbed(K, E, Norm, Ph_index, b, tau):
            dNdE = K / ((E/Norm)**(Ph_index + (b * np.log(E/Norm))))
            m_tau = -tau
            return dNdE * np.exp(m_tau)
        
#########################Maybie this can be moved before or later
    if Forward_folding:
        if Telescope == "CTAN_alpha":
            print("CTAN-alpha to be configured soon")
                
        elif Telescope == "MAGIC":
            Noffregions = 3
            def m2LogL(params):
                xdata = Etrue
                mtau = -tau
                if IRF_u:
                    mu_gam, mu_gam_u = dNdE_to_mu_MAGIC_IRF((fit_func(xdata, params) * np.exp(mtau * alpha)), Ebinsw_Etrue, migmatval, migmaterr, Eest)
                    mu_gam_final_u = mu_gam_u[minbin:maxbin]

                else:
                    mu_gam = dNdE_to_mu_MAGIC((fit_func(xdata, params) * np.exp(mtau * alpha)), Ebinsw_Etrue, migmatval, Eest)

                mu_gam_final = mu_gam[minbin:maxbin]
                Non_final = Non[minbin:maxbin] 
                Noff_final = Noff[minbin:maxbin]
                min_num_gauss = 20
            #####OLD not optimized way of doing this###########
                # if IRF_u:
                #     conditions = [((Non_final >= min_num_gauss) & (Noff_final >= min_num_gauss)), #change conditions and choices for irf
                #             (Non_final == 0.), 
                #             (Noff_final == 0.),
                #             (mu_gam_final < 1e-6),
                #             (mu_gam_final_u == 0),
                #             (Non_final != 0.) & (Noff_final != 0.)]
                #     choices = [Gauss_logL_IRF(Non_final, Noff_final, mu_gam_final, mu_gam_final_u, Noffregions),
                #             Poisson_logL_Non0_IRF(Non_final, Noff_final, mu_gam_final, mu_gam_final_u, Noffregions),
                #             Poisson_logL_Noff0_IRF(Non_final, Noff_final, mu_gam_final, mu_gam_final_u, Noffregions),
                #             Poisson_logL_small_mugam_IRF(Non_final, Noff_final, mu_gam_final, mu_gam_final_u, Noffregions),
                #             Poisson_logL_noIRF_IRF(Non_final, Noff_final, mu_gam_final, mu_gam_final_u, Noffregions),
                #             Poisson_logL_else_IRF(Non_final, Noff_final, mu_gam_final, mu_gam_final_u, Noffregions)]
                # else:
                #     conditions = [((Non_final >= min_num_gauss) & (Noff_final >= min_num_gauss)),
                #             (Non_final == 0.), 
                #             (Noff_final == 0.),
                #             (Non_final != 0.) & (Noff_final != 0.)]
                #     choices = [Gauss_logL(Non_final, Noff_final, mu_gam_final, Noffregions),
                #             Poisson_logL_Non0(Non_final, Noff_final, mu_gam_final, Noffregions),
                #             Poisson_logL_Noff0(Non_final, Noff_final, mu_gam_final, Noffregions),
                #             Poisson_logL_else(Non_final, Noff_final, mu_gam_final, Noffregions)]
                # res = np.select(conditions, choices, default = 999999999)
                
            #####NEW optimized way of doing this###########

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
            if fit_func_name == "MBPWL": #defines limits to faster and better find the minimum. Can be changed if the intrinsic spectrum function is changed.
                MBPWL_limits = ([(1e-6, 1e-3), (-4., 5.)])
                errors = ([1e-7, 0.01])
                for i in range(knots):
                    MBPWL_limits.append((0., 5.))
                    errors.append(0.01)
                m.limits = MBPWL_limits
            elif fit_func_name == "PWL":
                m.limits = ([(1e-7,1e-3), (None, 5.)])
                errors = [1e-7, 0.01]
            elif fit_func_name == "LP" or fit_func_name == "freeLP":
                m.limits = ([(1e-7, 1e-3), (None, 5.), (None, None)])
                errors = [1e-7, 0.01, 0.01]
            elif fit_func_name == "EPWL":
                m.limits = ([(1e-7, 1e-3), (None, 5.), (None, None)])
                errors = [1e-8, 1.0, np.sqrt(500.)]
            elif fit_func_name == "ELP":
                m.limits = ([(1e-7, 1e-3), (-2., 5.), (None, None), (None, None)])
                errors = [1e-8, 1., 0.1, np.sqrt(500.)]
            elif fit_func_name == "SEPWL":
                m.limits = ([(1e-7, 1e-3), (None, 5.), (None, None), (None, None)])
                errors = [1e-8, 1.0, np.sqrt(500.), 1.]
            elif fit_func_name == "SELP":
                m.limits = ([(1e-7, 1e-3), (-2., 5.), (None, None), (None, None), (None, None)])
                errors = [1e-8, 1., 0.1, np.sqrt(500.), 0.1]
            #TODO ADD other functions
            #m.tol = 1e-6
            #m.strategy = 2
            m.errors = errors
            
            m.migrad()
            return m

    else:
        raise Exception('The simulated spectrum "{func}" has not been implemented.'.format(func = Spectrum_func_name))
    
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
        migmatxEtrue = migrmatrix.axis("x").edges()/1e3 #TeV #edge values of X axis of the migration matrix (True Energy)
        migmatyEest = migrmatrix.axis("y").edges()/1e3 #TeV #edge values of Y axis of the migration matrix (Estimated Energy)

        Eest = migrmatrix.axis("y").centers()/1e3 #TeV #center values of X axis of the migration matrix (True Energy)
        Etrue = migrmatrix.axis("x").centers()/1e3 #TeV #center values of Y axis of the migration matrix (Estimated Energy)
        E_final = Etrue
        Usedbins = np.where((Emin <= Eest) & (Eest <= Emax))
        minbin = Usedbins[0][0]
        maxbin = Usedbins[0][-1] + 1
        Eest_final = Eest[minbin:maxbin]
        
        tau = tau_interp(Etrue, Source_z, EBL_Model, kind_of_interp = "log") #interpolate the tau values to have the same bins as the migration matrix and the data.
        Ebinsw_final = migmatyEest[1:] - migmatyEest[:-1] #compute the bin width of the final energy bins
        Ebinsw_Etrue = migmatxEtrue[1:] - migmatxEtrue[:-1] #compute the bin width of Etrue energy bins

        if Spectrum_func_name == "PWL":
            dNdEa = dNdE_absorbed(Source_flux, Etrue, Norm, Ph_index, tau) #use the previously defined dNdE function 

        elif Spectrum_func_name == "LP":
            dNdEa = dNdE_absorbed(Source_flux, Etrue, Norm, Ph_index, LP_curvature, tau) #use the previously defined dNdE function 

        mu_vec_final = dNdE_to_mu_MAGIC(dNdEa, Ebinsw_Etrue, migmatval, Eest) #get the dNdE to the needed mu values for the likelihood.

    else:
        raise Exception('The telescope "{func}" has not been implemented.'.format(func = Telescope))


    xdata = E_final
    datetime = time.strftime("%Y%m%d%H%M")
    iter = int(sys.argv[1])
        
    for fit_func_name in fit_n: #loop over the different fit functions
        print("Starting function {func} for iter {iter}".format(func = fit_func_name, iter = iter))
        EBL_Model, initial_guess_0, initial_guess_pos, step, last_bin, first_bin, knots, Efirst, DeltaE, Source_z = config_fit(fit_func_name)
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
                chisqs = []
                global alpha, Non, Noff
                alpha = initial_guess_pos
                rng_num = iter_num
                my_generator = np.random.default_rng(rng_num)
                Non, Noff = my_generator.poisson(mu_on), my_generator.poisson(Noffregions * mu_off)
                Non_u, Noff_u = np.sqrt(Non), np.sqrt(Noff)
                if Chain_guess:
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
                            print("Function {0} did not minimize properly the {1} intrinsic spectra for iteration {2}".format(fit_func_name, Spectrum_func_name, iter))
                            break
                            #print("Function {0} minimized properly the {1} intrinsic spectra for iteration {2}".format(fit_func_name, Spectrum_func_name, iter))
                        
                        if i < len(alphas):
                            initial_guess_mat[i+1] = things.values
                        chi2 = m2LogL(things.values)
                        chisqs.append(chi2)
                else:
                    for i, alpha0 in enumerate(alphas):
                        alpha = alpha0
                        things = fit(initial_guess = initial_guess_0)
                        if things.valid == False:
                            print("Function {0} did not minimize properly the {1} intrinsic spectra for iteration {2}".format(fit_func_name, Spectrum_func_name, iter))
                            break
                        chi2 = m2LogL(things.values)
                        chisqs.append(chi2)

                print("Function {0} minimized properly the {1} intrinsic spectra for iteration {2}".format(fit_func_name, Spectrum_func_name, iter))

                return chisqs

            alphas = alphas_creation(initial_guess_pos, first_bin, last_bin, step)

            my_generator2 = np.random.default_rng(iter)
            if systematics == 0.0:
                mu_on = mu_vec_final + bckgmu_final 
            else:
                mu_on = my_generator2.normal(mu_vec_final, systematics * mu_vec_final) + bckgmu_final #to add some systematics to try to fit the real results
            mu_off = bckgmu_final 

            chisqs = process2(iter, alphas, mu_on, mu_off)
            if len(chisqs) != len(alphas):
                continue
            if np.isnan(chisqs).any():
                continue
            dset = savefile.create_dataset("alphas", data = alphas, dtype='float')
            dset = savefile.create_dataset("chisqs", data = chisqs, dtype='float')

        else:
            print("Use Forward Folding please")