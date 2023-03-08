import numpy as np
import scipy as sc
from iminuit import Minuit
import pandas as pd
import scipy.interpolate as interpolate
from scipy.integrate import quad
from matplotlib import colors
from EBL_fit_MC_functions import *
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
Extratxt = "errors_IRF_test_fix"
#load all config from config file:
start_time = time.time()
with open("/data/magic/users-ifae/rgrau/EBL-splines/EBL_fit_config2_1.yml", "r") as f:
    inp_config = yaml.safe_load(f)
fit_func_name = inp_config["fit_func_name"]
Telescope = inp_config["Telescope"]
Spectrum_func_name = inp_config["Spectrum_func_name"]
EBL_Model = inp_config["EBL_Model"]
Source_flux = inp_config["Source_flux"]
Observation_time = inp_config["Observation_time"]
niter = inp_config["niter"]
Energy_migration = inp_config["Energy_migration"]
Background = inp_config["Background"]
Background_scale = inp_config["Background_scale"]
Forward_folding = inp_config["Forward_folding"]
initial_guess_pos = inp_config["initial_guess_pos"]
step = inp_config["step"]
last_bin = inp_config["last_bin"]
first_bin = inp_config["first_bin"]
Norm = inp_config["Norm"]
Ph_index = inp_config["Ph_index"]
initial_guess_0 = inp_config["initial_guess_0"]
LP_curvature = inp_config["LP_curvature"]
Source_z = inp_config["Source_z"]
IRF_u = inp_config["IRF_u"]

if fit_func_name == "MBPWL":
    knots = inp_config["knots"]
    Efirst = inp_config["Efirst"]
    DeltaE = inp_config["DeltaE"]
else:
    knots = 2
    Efirst = 0.11
    DeltaE = 0.306

#define fit function depending on the selected one in the configuration:
fit_func = fit_func_select(fit_func_name, knots, Efirst, DeltaE)

if Forward_folding:
    if Telescope == "CTAN_alpha": #this part needs to be changed to include the real CTAN_alpha configuration
        print("CTAN-alpha to be configured soon")
        # Noffregions = 5
        # def m2LogL(params):
        #     xdata = E_EBL
        #     mtau = -tau
        #     mu_gam0 = dNdE_to_mu((fit_func(xdata, params) * np.exp(mtau * alpha))[2:37], Effa_reb, Ebinsw[2:37], Observation_time, Ebins, Eres_reb2, E_EBL[2:37])
        #     mu_gam = mu_gam0[5:-4]
        #     mu_bg = mu_BG(mu_gam, Non, Noff, Noffregions)
        #     min_num_gauss = 20
        #     conditions = [((Non >= min_num_gauss) & (Noff >= min_num_gauss)), (Non == 0.), (Noff == 0.), (Non != 0.) & (Noff != 0.)]
        #     choices = [Gauss_logL(Non, Noff, mu_gam, Noffregions), Poisson_logL_Non0(Non, Noff, mu_gam, Noffregions), Poisson_logL_Noff0(Non, Noff, mu_gam, Noffregions), Poisson_logL(Non, Noff, mu_gam, mu_bg, Noffregions)]
        #     res = np.select(conditions, choices, default = 999999999)
        #     return np.sum(res)
            
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

            if IRF_u:
                conditions = [((Non_final >= min_num_gauss) & (Noff_final >= min_num_gauss)), #change conditions and choices for irf
                          (Non_final == 0.), 
                          (Noff_final == 0.),
                          (mu_gam_final < 1e-6),
                          (mu_gam_final_u == 0),
                          (Non_final != 0.) & (Noff_final != 0.)]
                choices = [Gauss_logL_IRF(Non_final, Noff_final, mu_gam_final, mu_gam_final_u, Noffregions),
                           Poisson_logL_Non0_IRF(Non_final, Noff_final, mu_gam_final, mu_gam_final_u, Noffregions),
                           Poisson_logL_Noff0_IRF(Non_final, Noff_final, mu_gam_final, mu_gam_final_u, Noffregions),
                           Poisson_logL_small_mugam_IRF(Non_final, Noff_final, mu_gam_final, mu_gam_final_u, Noffregions),
                           Poisson_logL_noIRF_IRF(Non_final, Noff_final, mu_gam_final, mu_gam_final_u, Noffregions),
                           Poisson_logL_else_IRF(Non_final, Noff_final, mu_gam_final, mu_gam_final_u, Noffregions)]
            else:
                conditions = [((Non_final >= min_num_gauss) & (Noff_final >= min_num_gauss)),
                          (Non_final == 0.), 
                          (Noff_final == 0.),
                          (Non_final != 0.) & (Noff_final != 0.)]
                choices = [Gauss_logL(Non_final, Noff_final, mu_gam_final, Noffregions),
                           Poisson_logL_Non0(Non_final, Noff_final, mu_gam_final, Noffregions),
                           Poisson_logL_Noff0(Non_final, Noff_final, mu_gam_final, Noffregions),
                           Poisson_logL_else(Non_final, Noff_final, mu_gam_final, Noffregions)]
            res = np.select(conditions, choices, default = 999999999)
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
            m.limits = ([(1e-7,1e-3), (None, None)])
            errors = [1e-7, 0.01]
        else:
            m.limits = ([(1e-7, 1e-3), (-2., None), (None, None)])
            errors = [1e-7, 0.01, 0.01]
        #m.tol = 1e-6
        #m.strategy = 2
        m.errors = errors
        
        m.migrad()
        return m

# else:
#     def LSQ(params):
#         return np.sum((ydata - fit_func(xdata, params)) ** 2 / ydata_u ** 2)

#     def fit(initial_guess):
#         LSQ.errordef = Minuit.LIKELIHOOD

#         m = Minuit(LSQ, initial_guess)

#         m.tol = 1e-20

#         m.migrad()
#         return m

# if EBL_Model == "Dominguez": #loads the EBL data of the Dominguez et al 2011 paper.
#     file = np.loadtxt('/data/magic/users-ifae/rgrau/EBL-splines/tau_dominguez11.out')
#     pdfile = pd.DataFrame(file)
#     pdfile = pdfile.rename(columns={ 0 : 'E [TeV]', 1: 'tau z=0.01', 2: 'tau z=0.02526316', 3: 'tau z=0.04052632', 4: 'tau z=0.05578947', 5: 'tau z=0.07105263', 6: 'tau z=0.08631579', 7: 'tau z=0.10157895', 8: 'tau z=0.11684211', 9: 'tau z=0.13210526', 10: 'tau z=0.14736842', 11: 'tau z=0.16263158', 12: 'tau z=0.17789474', 13: 'tau z=0.19315789', 14: 'tau z=0.20842105'})
#     E_EBL = pdfile['E [TeV]'].to_numpy() #energy bins
#     tau_EBL = pdfile['tau z=0.20842105'].to_numpy() #tau bins
# else:
#     raise Exception('The EBL model "{func}" has not been implemented.'.format(func = EBL_Model))

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

else:
    raise Exception('The simulated spectrum "{func}" has not been implemented.'.format(func = Spectrum_func_name))

if Telescope == "CTAN_alpha": #compute the values we need if the telescope selected is CTAN_alpha (needs to be changed as the Response function has to be changed)
    print("Not well implemented yet")
    # #Effective area:
    # fileEA = np.loadtxt('/data/magic/users-ifae/rgrau/EBL-splines/EffArea50h.txt', skiprows = 11)
    # EffA = pd.DataFrame(fileEA)
    # EffA = EffA.rename(columns={ 0 : 'E [TeV]', 1: 'Eff area (m^2)'})
    # EffaE= EffA['E [TeV]'].to_numpy()
    # Effa = EffA['Eff area (m^2)'].to_numpy()

    # Effa_reb = normal_interp1d(EffaE, Effa, E_EBL[2:37])


    # #Anglular resolution:
    # fileAng = np.loadtxt('/data/magic/users-ifae/rgrau/EBL-splines/Angres.txt', skiprows = 11)
    # Angresall = pd.DataFrame(fileAng)
    # Angresall = Angresall.rename(columns={ 0 : 'E [TeV]', 1: 'Angular resolution (deg)'})
    # AngresE = Angresall['E [TeV]'].to_numpy()
    # Angres = Angresall['Angular resolution (deg)'].to_numpy()
    # logE = np.log10(E_EBL)

    # Angres_reb = log_interp1d(AngresE, Angres, E_EBL[2:37])

    # #Background:
    # fileBkg = np.loadtxt('/data/magic/users-ifae/rgrau/EBL-splines/Bkg50h.txt', skiprows = 10)
    # Bkgpd = pd.DataFrame(fileBkg)
    # Bkgpd = Bkgpd.rename(columns={ 0 : 'E_min (TeV)', 1: 'E_max (TeV)', 2: 'Bck Rate (Hz/deg^2)'})

    # BckgEmin = Bkgpd['E_min (TeV)'].to_numpy()
    # BckgEmax = Bkgpd['E_max (TeV)'].to_numpy()
    # Bckg = Bkgpd['Bck Rate (Hz/deg^2)'].to_numpy()
    # BckgEdiff = BckgEmax - BckgEmin

    # BckgElogmean = np.power(10 ,np.mean([np.log10(BckgEmax), np.log10(BckgEmin)], axis = 0))

    # Ebinsmin = np.zeros(len(E_EBL))
    # Ebinsmax = np.zeros(len(E_EBL))

    # for i in range(1, len(E_EBL)):
    #     Ebinsmin[i] = 10**(np.mean([np.log10(E_EBL[i]), np.log10(E_EBL[i-1])]))
    # for i in range(len(E_EBL) - 1):
    #     Ebinsmax[i] = 10**(np.mean([np.log10(E_EBL[i]), np.log10(E_EBL[i+1])]))
        
    # Ebinsmin[0] = 10**(np.log10(Ebinsmin[1])-(np.log10(Ebinsmin[2])-np.log10(Ebinsmin[1])))
    # Ebinsmax[len(E_EBL)-1] = 10**(np.log10(Ebinsmax[len(E_EBL)-2])-(np.log10(Ebinsmax[len(E_EBL)-3])-np.log10(Ebinsmax[len(E_EBL)-2])))
    # Ebinsw = Ebinsmax - Ebinsmin
    # DifBckg = Bckg / (BckgEmax - BckgEmin) #Hz/deg**2

    # interpolation_bckg = log_interp1d2(BckgElogmean,DifBckg)

    # Bckg_reb = np.zeros([len(E_EBL[2:37]),2])
    # for i in range(len(E_EBL[2:37])):
    #     Bckg_reb[i] = quad(interpolation_bckg, Ebinsmin[i+2], Ebinsmax[i+2])
    # Bckg_reb = Bckg_reb[:,0]

    # #Treating the data:
    # skyang = (180/np.pi)**2 * 2 * np.pi * (1-np.cos(np.deg2rad(Angres_reb)))#deg^2
    # bckgmu = Bckg_reb * Observation_time * skyang

    # ##Generating dNdE_absorved and applying Eres to it
    # if Spectrum_func_name == "PWL":
    #     dNdEa = dNdE_absorbed(Source_flux, E_EBL, Norm, Ph_index, tau_EBL)

    # elif Spectrum_func_name == "LP":
    #     dNdEa = dNdE_absorbed(Source_flux, E_EBL, Norm, Ph_index, LP_curvature, tau_EBL)
    
    
    # logEbins = np.zeros(len(E_EBL[2:37])+1)
    # for i in range(len(E_EBL[2:37])+1):
    #     if i == 0:
    #         logEbins[i] = logE[2] - ((logE[3]-logE[2])/2)
    #     elif i == (len(E_EBL[2:37])):
    #         logEbins[i] = logE[2:37][i-1] + ((logE[2:37][i-1]-logE[2:37][i-2])/2)
    #     else:
    #         logEbins[i] = (logE[2:37][i] + logE[2:37][i-1]) / 2

    # Ebins = 10 ** logEbins
    # #Energy Resolution:

    # if Energy_migration:

    #     fileEres = np.loadtxt('/data/magic/users-ifae/rgrau/EBL-splines/Eres.txt', skiprows = 8)
    #     Eresall = pd.DataFrame(fileEres)
    #     Eresall = Eresall.rename(columns={ 0 : 'E [TeV]', 1: 'Energy resolution (deg)'})
    #     EresE = Eresall['E [TeV]'].to_numpy()
    #     Eres = Eresall['Energy resolution (deg)'].to_numpy()

    #     Eres_reb = log_interp1d(EresE[:-1], Eres[:-1], E_EBL[2:37])
    #     Eres_reb2 = Eres_reb * E_EBL[2:37]
        
    #     mu_vec_reco = dNdE_to_mu(dNdEa[2:37], Effa_reb, Ebinsw[2:37], Observation_time, Ebins, Eres_reb2, E_EBL[2:37])
    # else:
    #     mu_vec_reco = dNdEa[2:37] * Effa_reb * Ebinsw[2:37] * Observation_time

    # E_final = E_EBL[7:33] #[7:33]
    # mu_vec_final = mu_vec_reco[5:-4] #[5:-4]
    # bckgmu_final = bckgmu[5:-4] * Background_scale
    # Effa_final = Effa_reb[5:-4]
    # Ebinsw_final = Ebinsw[7:33]
    # tau = tau_EBL[7:33]

elif Telescope == "MAGIC": #compute values needed for minimization if the selected telescope is MAGIC
    
    Bckg = uproot.open("/data/magic/users-ifae/rgrau/EBL-splines/Output_flute.root:hEstBckgE")#load background values
    bckgmu_final = Bckg.values() #counts in 42480s (can be normalized for any time but as the migmatrix is for that time, only use that time).

    migrmatrix = uproot.open("/data/magic/users-ifae/rgrau/EBL-splines/fold_migmatrix.root:mig_matrix") #load migration matrix
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


# if Spectrum_func_name == 'LP':
#     if fit_func_name == "MBPWL":
#         if not os.path.exists('/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}({curv})_{func2}_{knots}knots_{telescope}'.format(curv = LP_curvature,func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope)):
#             os.mkdir('/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}({curv})_{func2}_{knots}knots_{telescope}'.format(curv = LP_curvature,func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope))
#         hdf5filename = "/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}({curv})_{func2}_{knots}knots_{telescope}/EBL_mult_nit{nit}of{niter}_{datetime}.hdf5".format(curv = LP_curvature,func1 = Spectrum_func_name, func2 = fit_func_name ,nit = iter, niter = niter, datetime = datetime, knots = knots, telescope = Telescope)
#         savefile = h5py.File(hdf5filename, "w")
#     else:
#         if not os.path.exists('/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}({curv})_{func2}_{telescope}'.format(curv = LP_curvature,func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope)):
#             os.mkdir('/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}({curv})_{func2}_{telescope}'.format(curv = LP_curvature,func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope))
#         hdf5filename = "/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}({curv})_{func2}_{telescope}/EBL_mult_nit{nit}of{niter}_{datetime}.hdf5".format(curv = LP_curvature,func1 = Spectrum_func_name, func2 = fit_func_name ,nit = iter, niter = niter, datetime = datetime, knots = knots, telescope = Telescope)
#         savefile = h5py.File(hdf5filename, "w")
# else:
#     if fit_func_name == "MBPWL":
#         if not os.path.exists('/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}_{func2}_{knots}knots_{telescope}'.format(func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope)):
#             os.mkdir('/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}_{func2}_{knots}knots_{telescope}'.format(func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope))
#         hdf5filename = "/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}_{func2}_{knots}knots_{telescope}/EBL_mult_nit{nit}of{niter}_{datetime}.hdf5".format(func1 = Spectrum_func_name, func2 = fit_func_name ,nit = iter, niter = niter, datetime = datetime, knots = knots, telescope = Telescope)
#         savefile = h5py.File(hdf5filename, "w")
#     else:
#         if not os.path.exists('/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}_{func2}_{telescope}'.format(func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope)):
#             os.mkdir('/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}_{func2}_{telescope}'.format(func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope))
#         hdf5filename = "/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}_{func2}_{telescope}/EBL_mult_nit{nit}of{niter}_{datetime}.hdf5".format(func1 = Spectrum_func_name, func2 = fit_func_name ,nit = iter, niter = niter, datetime = datetime, knots = knots, telescope = Telescope)
#         savefile = h5py.File(hdf5filename, "w")

# name the folder where the data will be stored and the datafile name
if Spectrum_func_name == "LP":
    if not os.path.exists('/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}{curv}_{func2}_{telescope}_with_{systematics}_Systematics_4w_fix_2{extra}'.format(curv = LP_curvature, func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt)):
        os.mkdir('/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}{curv}_{func2}_{telescope}_with_{systematics}_Systematics_4w_fix_2{extra}'.format(curv = LP_curvature, func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt))
        os.popen('cp /data/magic/users-ifae/rgrau/EBL-splines/EBL_fit_config2_1.yml /data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}{curv}_{func2}_{telescope}_with_{systematics}_Systematics_4w_fix_2{extra}/EBL_fit_config2_1.yml'.format(curv = LP_curvature, func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt))
    hdf5filename = "/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}{curv}_{func2}_{telescope}_with_{systematics}_Systematics_4w_fix_2{extra}/EBL_mult_nit{nit}of{niter}_{datetime}.hdf5".format(curv = LP_curvature, func1 = Spectrum_func_name, func2 = fit_func_name ,nit = iter, niter = niter, datetime = datetime, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt)
    savefile = h5py.File(hdf5filename, "w")
else:
    if not os.path.exists('/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}_{func2}_{telescope}_with_{systematics}_Systematics_4w_fix_2{extra}'.format(func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt)):
        os.mkdir('/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}_{func2}_{telescope}_with_{systematics}_Systematics_4w_fix_2{extra}'.format(func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt))
        os.popen('cp /data/magic/users-ifae/rgrau/EBL-splines/EBL_fit_config2_1.yml /data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}_{func2}_{telescope}_with_{systematics}_Systematics_4w_fix_2{extra}/EBL_fit_config2_1.yml'.format(func1 = Spectrum_func_name, func2 = fit_func_name, niter = niter, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt))
    hdf5filename = "/data/magic/users-ifae/rgrau/EBL-splines/EBL{niter}_{func1}_{func2}_{telescope}_with_{systematics}_Systematics_4w_fix_2{extra}/EBL_mult_nit{nit}of{niter}_{datetime}.hdf5".format(func1 = Spectrum_func_name, func2 = fit_func_name ,nit = iter, niter = niter, datetime = datetime, knots = knots, telescope = Telescope, systematics = systematics, extra = Extratxt)
    savefile = h5py.File(hdf5filename, "w")


#Do the forward folding
if Forward_folding:
    if Background != True:
        print("The forward folding is done with background")

    def process(alpha0): #this is not used, will have to delete it
        global alpha
        alpha = alpha0   
        things = fit(initial_guess = initial_guess_0)
        if things.valid == False:
            raise Warning("The minimum is not valid")
        return m2LogL(things.values)

    def process2(iter_num, alphas, mu_on, mu_off):
        chisqs = []
        global alpha, Non, Noff
        alpha = initial_guess_pos
        rng_num = iter_num
        my_generator = np.random.default_rng(rng_num)
        Non, Noff = my_generator.poisson(mu_on), my_generator.poisson(Noffregions * mu_off)
        Non_u, Noff_u = np.sqrt(Non), np.sqrt(Noff)
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
                raise Warning("The minimum is not valid")
            if i < len(alphas):
                initial_guess_mat[i+1] = things.values
            chi2 = m2LogL(things.values)
            chisqs.append(chi2)
        return chisqs

    alphas = alphas_creation(initial_guess_pos, first_bin, last_bin, step)

    my_generator2 = np.random.default_rng(iter)
    if systematics == 0.0:
        mu_on = mu_vec_final + bckgmu_final 
    else:
        mu_on = my_generator2.normal(mu_vec_final, systematics * mu_vec_final) + bckgmu_final #to add some systematics to try to fit the real results
    mu_off = bckgmu_final 

    chisqs = process2(iter, alphas, mu_on, mu_off)

    dset = savefile.create_dataset("alphas", data = alphas, dtype='float')
    dset = savefile.create_dataset("chisqs", data = chisqs, dtype='float')

else:
    print("Use Forward Folding please")
    # if Background: 
    #     ydata, ydata_u, dNdE_b, dNdE_b_u = SED_gen(15, bckgmu_final, mu_vec_final, Effa_final, Ebinsw_final, Observation_time, xdata, Noffregions)
    # else:
    #     ydata, ydata_u, dNdE_b, dNdE_b_u = SED_gen_nobckg(15, mu_vec_final, Effa_final, Ebinsw_final, Observation_time, xdata)

    # things = fit(initial_guess = initial_guess_0)

    # Egam = np.geomspace(E_final[1],E_final[-1], 300)

    # chisqs_iter = []
    # alphas_iter = []

    # for j in range(niter):
    #     if Background:
    #         SED, SED_u, dNdE_b, dNdE_b_u = SED_gen(j, bckgmu_final, mu_vec_final, Effa_final, Ebinsw_final, Observation_time, E_final)
    #     else:
    #         SED, SED_u, dNdE_b, dNdE_b_u = SED_gen_nobckg(j, mu_vec_final, Effa_final, Ebinsw_final, Observation_time, E_final)
    #     chisqs = []
    #     alphas = alphas_creation(initial_guess_pos, first_bin, last_bin, step)
    #     ydata, ydata_u = SED_alpha(initial_guess_pos, dNdE_b, dNdE_b_u, tau, E_final)
    #     things = fit(initial_guess=initial_guess_0)
    #     initial_guess_mat = ig_mat_create(fit_func_name, alphas, knots)
    #     initial_guess_mat[0] = things.values

    #     for i, alpha in enumerate(alphas):
    #         ydata, ydata_u = SED_alpha(alpha, dNdE_b, dNdE_b_u, tau, E_final)
    #         initial_guess = initial_guess_mat[i] #phi_0, lam1, deltas (len(deltas)=knots))
    #         if alpha == initial_guess_pos:
    #             initial_guess = initial_guess_mat[0]
    #         things = fit(initial_guess=initial_guess)
    #         if i < (len(alphas)):
    #             initial_guess_mat[i+1] = things.values
    #         chi2 = chisq(ydata, fit_func(E_final, things.values), ydata_u)
    #         chisqs.append(chi2)

    #     alphas_iter.append(alphas)
    #     chisqs_iter.append(chisqs)
    #     print("\033[2;31;43m iteration {iter} of {niter} finished \033[0;0m".format(iter = j+1, niter = niter))
        
    #     dset = savefile.create_dataset("alphas", data = alphas_iter, dtype='float')
    #     dset = savefile.create_dataset("chisqs", data = chisqs_iter, dtype='float')
