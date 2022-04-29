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

#load all config:
start_time = time.time()
with open("EBL_fit_config.yml", "r") as f:
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

if fit_func_name == "MBPWL":
        knots = inp_config["knots"]
        Efirst = inp_config["Efirst"]
        Elast = inp_config["Elast"]

#define fit function:
fit_func = fit_func_select(fit_func_name, knots, Efirst, Elast)

if Forward_folding:
    def m2LogL(params):
        xdata = E_EBL
        mtau = -tau_EBL
        mu_gam0 = dNdE_to_mu((fit_func(xdata, params) * np.exp(mtau * alpha))[2:37], Effa_reb, Ebinsw[2:37], Observation_time, Ebins, Eres_reb2, E_EBL[2:37])
        mu_gam = mu_gam0[5:-4]
        mu_bg = mu_BG(mu_gam, Non, Noff)
        min_num_gauss = 20
        conditions = [((Non >= min_num_gauss) & (Noff >= min_num_gauss)), (Non == 0.), (Noff == 0.), (Non != 0.) & (Noff != 0.)]
        choices = [Gauss_logL(Non, Noff, mu_gam), Poisson_logL_Non0(Non, Noff, mu_gam), Poisson_logL_Noff0(Non, Noff, mu_gam), Poisson_logL(Non, Noff, mu_gam, mu_bg)]
        res = np.select(conditions, choices, default = 999999999)
        return np.sum(res)

    def fit(initial_guess):
        
        m2LogL.errordef = Minuit.LIKELIHOOD
        m = Minuit(m2LogL, initial_guess)
        if fit_func_name == "MBPWL":
            m.limits = ([(1e-6,1e-3), (-4., 4.), (0., 4.), (0., 4.), (0., 4.)])
        elif fit_func_name == "PWL":
            m.limits = ([(1e-7,1e-3), (1, 3)])
        else:
            m.limits = ([(1e-7, 1e-3), (-2., None), (None, None)])
        m.migrad()
        return m

else:
    def LSQ(params):
        return np.sum((ydata - fit_func(xdata, params)) ** 2 / ydata_u ** 2)

    def fit(initial_guess):
        LSQ.errordef = Minuit.LIKELIHOOD

        m = Minuit(LSQ, initial_guess)

        m.tol = 1e-20

        m.migrad()
        return m

if EBL_Model == "Dominguez":
    file = np.loadtxt('tau_dominguez11.out')
    pdfile = pd.DataFrame(file)
    pdfile = pdfile.rename(columns={ 0 : 'E [TeV]', 1: 'tau z=0.01', 2: 'tau z=0.02526316', 3: 'tau z=0.04052632', 4: 'tau z=0.05578947', 5: 'tau z=0.07105263', 6: 'tau z=0.08631579', 7: 'tau z=0.10157895', 8: 'tau z=0.11684211', 9: 'tau z=0.13210526', 10: 'tau z=0.14736842', 11: 'tau z=0.16263158', 12: 'tau z=0.17789474', 13: 'tau z=0.19315789', 14: 'tau z=0.20842105'})
    E_EBL = pdfile['E [TeV]'].to_numpy()
    tau_EBL = pdfile['tau z=0.20842105'].to_numpy()
else:
    raise Exception('The EBL model "{func}" has not been implemented.'.format(func = EBL_Model))

if Spectrum_func_name == "PWL":
    def dNdE_absorbed(K, E, Norm, Ph_index):
        dNdE = K / ((E/Norm)**Ph_index)
        m_tau = -tau_EBL
        return dNdE * np.exp(m_tau)

elif Spectrum_func_name == "LP":
    print("Not yet implemented")

else:
    raise Exception('The simulated spectrum "{func}" has not been implemented.'.format(func = Spectrum_func_name))

if Telescope == "CTAN_alpha":
    
    #Effective area:
    fileEA = np.loadtxt('EffArea50h.txt', skiprows = 11)
    EffA = pd.DataFrame(fileEA)
    EffA = EffA.rename(columns={ 0 : 'E [TeV]', 1: 'Eff area (m^2)'})
    EffaE= EffA['E [TeV]'].to_numpy()
    Effa = EffA['Eff area (m^2)'].to_numpy()

    Effa_reb = normal_interp1d(EffaE, Effa, E_EBL[2:37])


    #Anglular resolution:
    fileAng = np.loadtxt('Angres.txt', skiprows = 11)
    Angresall = pd.DataFrame(fileAng)
    Angresall = Angresall.rename(columns={ 0 : 'E [TeV]', 1: 'Angular resolution (deg)'})
    AngresE = Angresall['E [TeV]'].to_numpy()
    Angres = Angresall['Angular resolution (deg)'].to_numpy()
    logE = np.log10(E_EBL)

    Angres_reb = log_interp1d(AngresE, Angres, E_EBL[2:37])

    #Background:
    fileBkg = np.loadtxt('Bkg50h.txt', skiprows = 10)
    Bkgpd = pd.DataFrame(fileBkg)
    Bkgpd = Bkgpd.rename(columns={ 0 : 'E_min (TeV)', 1: 'E_max (TeV)', 2: 'Bck Rate (Hz/deg^2)'})

    BckgEmin = Bkgpd['E_min (TeV)'].to_numpy()
    BckgEmax = Bkgpd['E_max (TeV)'].to_numpy()
    Bckg = Bkgpd['Bck Rate (Hz/deg^2)'].to_numpy()
    BckgEdiff = BckgEmax - BckgEmin

    BckgElogmean = np.power(10 ,np.mean([np.log10(BckgEmax), np.log10(BckgEmin)], axis = 0))

    Ebinsmin = np.zeros(len(E_EBL))
    Ebinsmax = np.zeros(len(E_EBL))

    for i in range(1, len(E_EBL)):
        Ebinsmin[i] = 10**(np.mean([np.log10(E_EBL[i]), np.log10(E_EBL[i-1])]))
    for i in range(len(E_EBL) - 1):
        Ebinsmax[i] = 10**(np.mean([np.log10(E_EBL[i]), np.log10(E_EBL[i+1])]))
        
    Ebinsmin[0] = 10**(np.log10(Ebinsmin[1])-(np.log10(Ebinsmin[2])-np.log10(Ebinsmin[1])))
    Ebinsmax[len(E_EBL)-1] = 10**(np.log10(Ebinsmax[len(E_EBL)-2])-(np.log10(Ebinsmax[len(E_EBL)-3])-np.log10(Ebinsmax[len(E_EBL)-2])))
    Ebinsw = Ebinsmax - Ebinsmin
    DifBckg = Bckg / (BckgEmax - BckgEmin) #Hz/deg**2

    interpolation_bckg = log_interp1d2(BckgElogmean,DifBckg)

    Bckg_reb = np.zeros([len(E_EBL[2:37]),2])
    for i in range(len(E_EBL[2:37])):
        Bckg_reb[i] = quad(interpolation_bckg, Ebinsmin[i+2], Ebinsmax[i+2])
    Bckg_reb = Bckg_reb[:,0]

    #Treating the data:
    skyang = (180/np.pi)**2 * 2 * np.pi * (1-np.cos(np.deg2rad(Angres_reb)))#deg^2
    bckgmu = Bckg_reb * Observation_time * skyang

    ##Generating dNdE_absorved and applying Eres to it
    dNdEa = dNdE_absorbed(Source_flux, E_EBL, Norm, Ph_index)
    
    logEbins = np.zeros(len(E_EBL[2:37])+1)
    for i in range(len(E_EBL[2:37])+1):
        if i == 0:
            logEbins[i] = logE[2] - ((logE[3]-logE[2])/2)
        elif i == (len(E_EBL[2:37])):
            logEbins[i] = logE[2:37][i-1] + ((logE[2:37][i-1]-logE[2:37][i-2])/2)
        else:
            logEbins[i] = (logE[2:37][i] + logE[2:37][i-1]) / 2

    Ebins = 10 ** logEbins
    #Energy Resolution:

    if Energy_migration:

        fileEres = np.loadtxt('Eres.txt', skiprows = 8)
        Eresall = pd.DataFrame(fileEres)
        Eresall = Eresall.rename(columns={ 0 : 'E [TeV]', 1: 'Energy resolution (deg)'})
        EresE = Eresall['E [TeV]'].to_numpy()
        Eres = Eresall['Energy resolution (deg)'].to_numpy()

        Eres_reb = log_interp1d(EresE[:-1], Eres[:-1], E_EBL[2:37])
        Eres_reb2 = Eres_reb * E_EBL[2:37]
        
        mu_vec_reco = dNdE_to_mu(dNdEa[2:37], Effa_reb, Ebinsw[2:37], Observation_time, Ebins, Eres_reb2, E_EBL[2:37])
    else:
        mu_vec_reco = dNdEa[2:37] * Effa_reb * Ebinsw[2:37] * Observation_time

    E_final = E_EBL[7:33] #[7:33]
    mu_vec_final = mu_vec_reco[5:-4] #[5:-4]
    bckgmu_final = bckgmu[5:-4] * Background_scale
    Effa_final = Effa_reb[5:-4]
    Ebinsw_final = Ebinsw[7:33]
    tau = tau_EBL[7:33]

elif Telescope == "MAGIC":
    print("MAGIC not implemented yet")

else:
    raise Exception('The telescope "{func}" has not been implemented.'.format(func = Telescope))


xdata = E_final
datetime = time.strftime("%Y%m%d%H%M%S")
hdf5filename = "EBL_fit_MC_{data}_nit{nit}_{datetime}.hdf5".format(data=fit_func_name, nit = niter, datetime = datetime)
savefile = h5py.File(hdf5filename, "w")

if Forward_folding:
    if Background != True:
        print("The forward folding is done with background")

    def process(alpha0):
        global alpha
        alpha = alpha0   
        things = fit(initial_guess = initial_guess_0)
        if things.valid == False:
            raise Warning("The minimum is not valid")
        # if alpha == 1:
        #     print("The best fit params for alpha = 1 are: ", things.values)
        return m2LogL(things.values)

    def process2(iter_num, alphas, mu_on, mu_off):
        print("\033[2;31;43m iteration {iter_num} started \033[0;0m".format(iter_num = iter_num))
        chisqs = []
        global alpha, Non, Noff
        alpha = initial_guess_pos
        rng_num = iter_num
        my_generator = np.random.default_rng(rng_num)
        Non, Noff = my_generator.poisson(mu_on), my_generator.poisson(5 * mu_off)
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
        print("\033[2;31;43m iteration {iter_num} finished \033[0;0m".format(iter_num = iter_num))
        return chisqs

    if niter == 1:
        if fit_func_name == "MBPWL":
                
            alphas = alphas_creation(initial_guess_pos, first_bin, last_bin, step)
            chisqs = []

            mu_on = mu_vec_final + bckgmu_final
            mu_off = bckgmu_final 

            chisqs = process2(231654, alphas, mu_on, mu_off)

            # rng_num = 231654
            # my_generator = np.random.default_rng(rng_num)
            # Non, Noff = my_generator.poisson(mu_on), my_generator.poisson(5 * mu_off)
            # Non_u, Noff_u = np.sqrt(Non), np.sqrt(Noff)
            # chisqs = Parallel(n_jobs=5)(delayed(process)(alpha0) for alpha0 in alphas)

            print("The minimum for the -2logL is at: ", alphas[np.where(chisqs == min(chisqs))])
            plt.plot(alphas, chisqs, 'o')
            # plt.yscale('log')
            plt.xlabel(r'$\alpha$')
            plt.ylabel(r'$-2log(L/L_{max})$')
            plt.show()

        else:
            alphas = alphas_creation(initial_guess_pos, first_bin, last_bin, step)
            chisqs = []

            mu_on = mu_vec_final + bckgmu_final
            mu_off = bckgmu_final 

            rng_num = 231654
            my_generator = np.random.default_rng(rng_num)
            Non, Noff = my_generator.poisson(mu_on), my_generator.poisson(5 * mu_off)
            Non_u, Noff_u = np.sqrt(Non), np.sqrt(Noff)
            chisqs = Parallel(n_jobs=-1)(delayed(process)(alpha0) for alpha0 in alphas)

            print("The minimum for the -2logL is at: ", alphas[np.where(chisqs == min(chisqs))])
            plt.plot(alphas, chisqs, 'o')
            # plt.yscale('log')
            plt.xlabel(r'$\alpha$')
            plt.ylabel(r'$-2log(L/L_{max})$')
            plt.show()

    else:
        alphas = alphas_creation(initial_guess_pos, first_bin, last_bin, step)
        alphas_iter = np.tile(alphas, [niter,1])
        chisqs_iter = []

        mu_on = mu_vec_final + bckgmu_final
        mu_off = bckgmu_final 

        chisqs_iter = Parallel(n_jobs=-1)(delayed(process2)(random_num, alphas, mu_on, mu_off) for random_num in range(niter))

        dset = savefile.create_dataset("alphas", data = alphas_iter, dtype='float')
        dset = savefile.create_dataset("chisqs", data = chisqs_iter, dtype='float')

else:

    if Background: 
        ydata, ydata_u, dNdE_b, dNdE_b_u = SED_gen(15, bckgmu_final, mu_vec_final, Effa_final, Ebinsw_final, Observation_time, xdata)
    else:
        ydata, ydata_u, dNdE_b, dNdE_b_u = SED_gen_nobckg(15, mu_vec_final, Effa_final, Ebinsw_final, Observation_time, xdata)

    things = fit(initial_guess = initial_guess_0)

    Egam = np.geomspace(E_final[1],E_final[-1], 300)

    chisqs_iter = []
    alphas_iter = []

    for j in range(niter):
        if Background:
            SED, SED_u, dNdE_b, dNdE_b_u = SED_gen(j, bckgmu_final, mu_vec_final, Effa_final, Ebinsw_final, Observation_time, E_final)
        else:
            SED, SED_u, dNdE_b, dNdE_b_u = SED_gen_nobckg(j, mu_vec_final, Effa_final, Ebinsw_final, Observation_time, E_final)
        chisqs = []
        alphas = alphas_creation(initial_guess_pos, first_bin, last_bin, step)
        ydata, ydata_u = SED_alpha(initial_guess_pos, dNdE_b, dNdE_b_u, tau, E_final)
        things = fit(initial_guess=initial_guess_0)
        initial_guess_mat = ig_mat_create(fit_func_name, alphas, knots)
        initial_guess_mat[0] = things.values

        for i, alpha in enumerate(alphas):
            ydata, ydata_u = SED_alpha(alpha, dNdE_b, dNdE_b_u, tau, E_final)
            initial_guess = initial_guess_mat[i] #phi_0, lam1, deltas (len(deltas)=knots))
            if alpha == initial_guess_pos:
                initial_guess = initial_guess_mat[0]
            things = fit(initial_guess=initial_guess)
            if i < (len(alphas)):
                initial_guess_mat[i+1] = things.values
            chi2 = chisq(ydata, fit_func(E_final, things.values), ydata_u)
            chisqs.append(chi2)

        alphas_iter.append(alphas)
        chisqs_iter.append(chisqs)
        print("\033[2;31;43m iteration {iter} of {niter} finished \033[0;0m".format(iter = j+1, niter = niter))
        
        dset = savefile.create_dataset("alphas", data = alphas_iter, dtype='float')
        dset = savefile.create_dataset("chisqs", data = chisqs_iter, dtype='float')

print("--- %m seconds ---" % (time.time() - start_time))