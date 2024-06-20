import numpy as np
import scipy.interpolate as interpolate
from scipy.integrate import quad
from scipy.stats import poisson, norm
import pandas as pd
import yaml
from ebltable.tau_from_model import OptDepth
from pathlib import Path

filepath = Path(__file__).resolve().parent
pathstring = str(filepath) + "/"
Norm = 0.25 #normalization energy in TeV

def general_config():
    with open("{0}config/EBL_MC_general_config.yml".format(pathstring), "r") as f:
        inp_config = yaml.safe_load(f)
    niter = inp_config["niter"]
    Energy_migration = inp_config["Energy_migration"]
    Forward_folding = inp_config["Forward_folding"]
    IRF_u = inp_config["IRF_u"]    
    Background = inp_config["Background"]
    fit_func_name = inp_config["fit_func_name"]
    Spectrum_func_name = inp_config["Spectrum_func_name"]
    Telescope = inp_config["Telescope"]
    return Telescope, niter, Energy_migration, Forward_folding, IRF_u, Background, fit_func_name, Spectrum_func_name
    
#load all config from config file:
def config_data(Spectrum_func_name):
    with open("{0}config/EBL_MC_config_data_{1}.yml".format(pathstring, Spectrum_func_name), "r") as f:
        inp_config = yaml.safe_load(f)
    #fit_func_name = inp_config["fit_func_name"]
    
    #Spectrum_func_name = inp_config["Spectrum_func_name"]
    Source_flux = inp_config["Source_flux"]
    Observation_time = inp_config["Observation_time"]
    Background_scale = inp_config["Background_scale"]
    Norm = inp_config["Norm"]
    Ph_index = inp_config["Ph_index"]
    LP_curvature = inp_config["LP_curvature"]
    Source_z = inp_config["Source_z"]
    E_cut = inp_config["E_cut"]
    d = inp_config["d"]
    EBL_Model_sim = inp_config["EBL_Model_sim"]

    return Source_flux, Observation_time, Background_scale, Norm, Ph_index, LP_curvature, E_cut, d, Source_z, EBL_Model_sim

 
def config_fit(fit_func_name):
    with open("{0}config/EBL_MC_config_fit_{1}.yml".format(pathstring, fit_func_name), "r") as f:
        inp_config = yaml.safe_load(f)
    EBL_Model_fit = inp_config["EBL_Model_fit"]
    initial_guess_0 = inp_config["initial_guess_0"]
    step = inp_config["step"]
    last_bin = inp_config["last_bin"]
    first_bin = inp_config["first_bin"]
    Source_z = inp_config["Source_z"]
    
    if fit_func_name == "MBPWL":
        knots = inp_config["knots"]
        Efirst = inp_config["Efirst"]
        DeltaE = inp_config["DeltaE"]
    else:
        knots = 2
        Efirst = 0.11
        DeltaE = 0.306

    return EBL_Model_fit, initial_guess_0, step, last_bin, first_bin, knots, Efirst, DeltaE, Source_z
    
def fit_func_select(fit_func_name, knots = 3, Efirst = 0.2 , DeltaE = 1.12):
    if fit_func_name == "MBPWL":
        def fit_func(xdata, params):
            if knots == 1:
                polw = np.zeros(len(xdata))
                gamma = np.zeros(knots+1)
                phi = np.zeros(knots+1)
                phi_0 = params[0] #len(sqrtdelta_lam) = len(lam)-1 = len(phi)-1
                gamma0 = params[1]
                sqrtdelta_gamma = params[2:knots+2]
                Eknot = Eknot
                delta_gamma = np.square(sqrtdelta_gamma)
                gamma[0] = gamma0
                phi[0] = phi_0
                gamma[1] = gamma[0] + delta_gamma[0]
                phi[1] = phi[0] * (Eknot/Norm) ** delta_gamma[0]
                for i in range(len(xdata)):
                    if xdata[i] < Eknot:
                        polw[i] = phi[0] * (xdata[i]/Norm) ** (-gamma[0])
                    elif xdata[i] >= Eknot:
                        polw[i] = phi[1] * (xdata[i]/Norm) ** (-gamma[1])
                return polw
            else:
                polw = np.zeros(len(xdata))

                gamma = np.zeros(knots+1)
                phi = np.zeros(knots+1)
                phi_0 = params[0] #len(sqrtdelta_lam) = len(lam)-1 = len(phi)-1
                gamma0 = params[1]
                sqrtdelta_gamma = params[2:knots+2]
                Elast = Efirst + DeltaE
                Ebr = np.geomspace(Efirst, Elast, knots)
                delta_gamma = np.square(sqrtdelta_gamma)
                gamma[0] = gamma0
                phi[0] = phi_0
                for i in range(knots):
                    gamma[i+1] = gamma[i] + delta_gamma[i]
                    phi[i+1] = phi[i] * (Ebr[i]/Norm) ** delta_gamma[i]
                for i in range(len(xdata)):
                    for j in range(knots):
                        if xdata[i]<Ebr[0]:
                            polw[i] = phi[0] * (xdata[i]/Norm) ** (-gamma[0])
                        elif Ebr[-1] < xdata[i]:
                            polw[i] = phi[-1] * (xdata[i]/Norm) ** (-gamma[-1])
                        elif Ebr[j] <= xdata[i] < Ebr[j+1]:
                            polw[i] = phi[j+1] * (xdata[i]/Norm) ** (-gamma[j+1])
                return polw
        return(fit_func)

    elif fit_func_name == "PWL":
        def fit_func(xdata, params):
            phi = params[0]
            gamma = params[1]
            PLW = phi / ((xdata/Norm) ** gamma)
            return PLW
        return(fit_func)

    elif fit_func_name == "LP":
        def fit_func(xdata, params):
            phi0 = params[0]
            alpha = params[1]
            beta = params[2]
            #Enorm = 1TeV  # (it is 0.249 TeV set by default) if it is 1 TeV no need ot include it (if it is different, need to add it to te LP function)
            LP = phi0 * np.power((xdata/Norm), (-alpha - beta * beta * np.log10(xdata/Norm)))
            return LP
        return(fit_func)

    elif fit_func_name == "freeLP":
        def fit_func(xdata, params):
            phi0 = params[0]
            alpha = params[1]
            beta = params[2]
            #Enorm = 1TeV #if it is 1 TeV no need ot include it (if it is different, need to add it to te LP function)
            freeLP = phi0 * np.power((xdata/Norm), (-alpha - beta * abs(beta) * np.log10(xdata/Norm)))
            return freeLP
        return(fit_func)
    
    elif fit_func_name == "EPWL":
        def fit_func(xdata, params):
            phi = params[0]
            gamma = params[1]
            c = params[2]
            EPWL = phi / ((xdata/Norm) ** gamma) * np.exp(-xdata / (c * c))
            return EPWL
        return fit_func
    
    elif fit_func_name == "ELP":
        def fit_func(xdata, params):
            phi0 = params[0]
            alpha = params[1]
            beta = params[2]
            c = params[3]
            ELP = phi0 * np.power((xdata/Norm), (-alpha - beta * beta * np.log10(xdata/Norm))) * np.exp(-xdata / (c * c))
            return ELP
        return fit_func
        
    elif fit_func_name == "SEPWL":
        def fit_func(xdata, params):
            phi = params[0]
            gamma = params[1]
            c = params[2]
            d = params[3]
            SEPWL = phi / ((xdata/Norm) ** gamma) * np.exp(-np.power(xdata / (c * c), d))
            return SEPWL
        return fit_func
    
    elif fit_func_name == "SELP":
        def fit_func(xdata, params):
            phi0 = params[0]
            alpha = params[1]
            beta = params[2]
            c = params[3]
            d = params[4]
            SELP = phi0 * np.power((xdata/Norm), (-alpha - beta * beta * np.log10(xdata/Norm))) * np.exp(-np.power(xdata / (c * c), d))
            return SELP
        return fit_func

    else:
        raise Exception('The function "{func}" has not been implemented.'.format(func = fit_func_name))

def alphas_creation(initial_guess_pos, first_bin, last_bin, step):
    alphas1 = np.arange(initial_guess_pos + step, last_bin + step, step)
    alphas2 = np.arange(initial_guess_pos, first_bin - step, -step)
    alphas = np.append(alphas1, alphas2)
    return np.round(alphas, decimals = 3)

def ig_mat_create(fit_func_name, alphas, knots):
    if fit_func_name == "MBPWL":
        initial_guess_mat = np.zeros((len(alphas)+1, knots+2))
    elif fit_func_name == "PWL":
        initial_guess_mat = np.zeros((len(alphas)+1, 2))
    elif fit_func_name == "LP" or fit_func_name == "freeLP":
        initial_guess_mat = np.zeros((len(alphas)+1, 3))
    elif fit_func_name == "EPWL":
        initial_guess_mat = np.zeros((len(alphas)+1, 3))
    elif fit_func_name == "ELP":
        initial_guess_mat = np.zeros((len(alphas)+1, 4))
    elif fit_func_name == "SEPWL":
        initial_guess_mat = np.zeros((len(alphas)+1, 4))
    elif fit_func_name == "SELP":
        initial_guess_mat = np.zeros((len(alphas)+1, 5))
    return initial_guess_mat

def mu_BG(mu_g, Non, Noff, Noffregions):
    mubg = ((-(Noffregions+1) * mu_g) + Non + Noff + np.sqrt(np.square(((Noffregions+1) * mu_g) - Non - Noff) + (4*(Noffregions+1) * Noff * mu_g)))/(2*(Noffregions+1))
    return mubg

def dNdE_to_mu(dNdEa, Ebinw, migmatval, Eest):
    mu_vec = dNdEa * Ebinw
    mu_vec_reco = np.zeros(len(Eest))
    for i in range(len(Eest)):
        mu_vec_reco[i] = np.sum(mu_vec * migmatval[:,i])
    return mu_vec_reco

def dNdE_to_mu_Eshift(dNdEa, Ebinw, migmatval, Eest, shift, seed, Etrue):
    migmat2 = np.zeros(migmatval.shape)
    np.random.seed(seed)
    Ratio = Etrue[1]/Etrue[0]
    migmat_shft_percent = np.abs(np.random.normal(.0, shift, migmat2[:,0].shape))
    migmat_shft = (np.log(1+migmat_shft_percent)/np.log(Ratio)).astype(int)
    for i, g in enumerate(migmat_shft):
        if i-g < 0:
            migmat2[i] = migmatval[i] * 0
        else:
            migmat2[i] = migmatval[i-g]
    mu_vec = dNdEa * Ebinw 
    mu_vec_reco = np.zeros(len(Eest))
    for i in range(len(Eest)):#canviar index de manera aleatoria
        mu_vec_reco[i] = np.sum(mu_vec * migmat2[:,i])
    return mu_vec_reco

def best_mubg_mugam_IRF(Non, Noff, mu_gam, delta_mu_gam, Noffregions):
    def mu_gam_f(eps, mu_gam, delta_mu_gam):
        return mu_gam + eps * delta_mu_gam
    def mu_BG_2_deq(alpha, Noff, Non, mu_gam):
        a = alpha + 1
        b = (1 + alpha) * mu_gam - alpha * (Non + Noff)
        c = -alpha * Noff * mu_gam
        mubg = (-b + np.sqrt(np.square(b) - 4 * a * c)) / (2. * a)
        return mubg
    
    fAlpha = 1/Noffregions
    a = -fAlpha
    b = delta_mu_gam * (1 - fAlpha) - mu_gam/delta_mu_gam * fAlpha
    c = fAlpha * (Non + Noff) + np.square(delta_mu_gam) + mu_gam * (1 - fAlpha)
    d = delta_mu_gam * (mu_gam + fAlpha * Noff - Non)
    # if np.isnan(mu_gam) or np.isnan(delta_mu_gam):
    #     print("mu_gam or delta_mu_gam is nan")
    #     return np.nan, np.nan
    # else:
    try:
        epsilon = np.roots([a, b, c, d])
        epsilon2 = np.real(epsilon[np.isreal(epsilon)])
        chi2proxy = np.zeros(len(epsilon2))
        mu_gam2 = np.zeros(len(epsilon2))
        mu_bg2 = np.zeros(len(epsilon2))
        for i, eps in enumerate(epsilon2):
            mu_gam2[i] = mu_gam_f(eps, mu_gam, delta_mu_gam)
            mu_bg2[i] = mu_BG_2_deq(fAlpha, Noff, Non, mu_gam2[i])
            chi2proxy[i] = -2*(np.log(poisson.pmf(Non, mu_bg2[i]+mu_gam2[i])) + np.log(poisson.pmf(Noff, mu_bg2[i]/fAlpha)) - 0.5*eps*eps)
        best_mugam, best_mubg = mu_gam2[np.argmin(chi2proxy)], mu_bg2[np.argmin(chi2proxy)]

    except:
        best_mugam = np.nan
        best_mubg = np.nan
    return best_mugam, best_mubg

def dNdE_to_mu_IRF(dNdEa, Ebinw, migmatval, migmaterr, Eest):
    mu_vec = dNdEa * Ebinw
    mu_vec_reco = np.zeros(len(Eest))
    mu_vec_reco_u = np.zeros(len(Eest))
    for i in range(len(Eest)):
        mu_vec_reco[i] = np.sum(mu_vec * migmatval[:,i])
        mu_vec_reco_u[i] = np.sqrt(np.sum(mu_vec*mu_vec * np.square(migmaterr[:,i]))) #as mu_vec_u = 0
    return mu_vec_reco, mu_vec_reco_u

def dNdE_to_mu_IRF_Eshift(dNdEa, Ebinw, migmatval, migmaterr, Eest, shift, seed, Etrue): #no needed
    migmat2 = np.zeros(migmatval.shape)
    migmaterr2 = np.zeros(migmaterr.shape)
    np.random.seed(seed)
    Ratio = Etrue[1]/Etrue[0]
    migmat_shft_percent = np.abs(np.random.normal(.0, shift, migmat2[:,0].shape))
    migmat_shft = (np.log(1+migmat_shft_percent)/np.log(Ratio)).astype(int)
    for i, g in enumerate(migmat_shft):
        if i-g < 0:
            migmat2[i] = migmatval[i] * 0
            migmaterr2[i] = migmaterr[i] * 0
        else:
            migmat2[i] = migmatval[i-g]
            migmaterr2[i] = migmaterr[i-g]
    mu_vec = dNdEa * Ebinw 
    mu_vec_reco = np.zeros(len(Eest))
    mu_vec_reco_u = np.zeros(len(Eest))
    for i in range(len(Eest)):#canviar index de manera aleatoria
        mu_vec_reco[i] = np.sum(mu_vec * migmat2[:,i])
        mu_vec_reco_u[i] = np.sqrt(np.sum(mu_vec*mu_vec * np.square(migmaterr[:,i]))) #as mu_vec_u = 0
    return mu_vec_reco, mu_vec_reco_u

def Poisson_logL_IRF(Non, Noff, mu_gam, mu_gam2, delta_mu_gam, mu_bg, Noffregions): #expectedgammas = mu_gam, bckg = Noff/Noffregions, observed = Non
    mugamma_gauss = norm.pdf(mu_gam2, mu_gam, delta_mu_gam)

    logL = np.log(poisson.pmf(Non, mu_gam2 + mu_bg)) + np.log(poisson.pmf(Noff, Noffregions * mu_bg)) + np.log(mugamma_gauss)
    logLmax = np.log(poisson.pmf(Non, Non)) + np.log(poisson.pmf(Noff, Noff)) + np.log(norm.pdf(mu_gam, mu_gam, delta_mu_gam))

    return -2 * (logL - logLmax)

def Poisson_logL_Non0_IRF(Non, Noff, mu_gam, delta_mu_gam, Noffregions): #canviat per IRF
    mu_bg = Noff / (1. + Noffregions)
    mu_gam2 = -np.square(delta_mu_gam) + mu_gam
    if mu_gam2 < 0.:#FIXME?
        mu_gam2 = 0
    return Poisson_logL_IRF(Non, Noff, mu_gam, mu_gam2, delta_mu_gam, mu_bg, Noffregions)

def Poisson_logL_Noff0_IRF(Non, Noff, mu_gam, delta_mu_gam, Noffregions):
    fAlpha = 1/Noffregions
    mu_bg = fAlpha * Non / (1 + fAlpha) - mu_gam -np.square(delta_mu_gam)/fAlpha
    if mu_bg < 0.:
        mu_bg = 0
        a = 1.
        b = -mu_gam + np.square(delta_mu_gam)
        c = -Non * np.square(delta_mu_gam)
        mu_gam2 = (-b + np.sqrt(np.square(b) - 4 * a * c)) / (2. * a)
    else:
        mu_gam2 = mu_gam + np.square(delta_mu_gam) / fAlpha
    return Poisson_logL_IRF(Non, Noff, mu_gam, mu_gam2, delta_mu_gam, mu_bg, Noffregions)

def Poisson_logL_small_mugam_IRF(Non, Noff, mu_gam, delta_mu_gam, Noffregions):
    fAlpha = 1/Noffregions
    mu_bg = fAlpha * (Noff + Non) / (1+fAlpha)
    return Poisson_logL_IRF(Non, Noff, mu_gam, mu_gam, delta_mu_gam, mu_bg, Noffregions)

def Poisson_logL_noIRF_IRF(Non, Noff, mu_gam, delta_mu_gam, Noffregions):
    return Poisson_logL_else(Non, Noff, mu_gam, Noffregions)

def Poisson_logL_else_IRF(Non, Noff, mu_gam, delta_mu_gam, Noffregions):
    mu_gam2, mu_bg = best_mubg_mugam_IRF(Non, Noff, mu_gam, delta_mu_gam, Noffregions)
    # if np.isnan(mu_gam2):
    #     return 99999999999999
    if mu_gam2 < 0.: #elif
        mu_gam2 = 0
        mu_bg = (Non + Noff)/ (1 + Noffregions)
    
    return Poisson_logL_IRF(Non, Noff, mu_gam, mu_gam2, delta_mu_gam, mu_bg, Noffregions)

def Gauss_logL_IRF(Non, Noff, mu_gam, delta_mu_gam, Noffregions): #canviat per IRF
    diff = Non - Noff/Noffregions - mu_gam
    delta_exp = np.square(delta_mu_gam) + Noff/(Noffregions*Noffregions)
    delta_diff = Non + delta_exp
    
    return np.square(diff)/delta_diff


def Poisson_logL(Non, Noff, mu_gam, mu_bg, Noffregions):
    logL = np.log(poisson.pmf(Non, mu_gam + mu_bg)) + np.log(poisson.pmf(Noff, Noffregions * mu_bg))
    logLmax = np.log(poisson.pmf(Non, Non)) + np.log(poisson.pmf(Noff, Noff))
    return -2 * (logL - logLmax)

def Poisson_logL_Non0(Non, Noff, mu_gam, Noffregions):
    mu_bg = Noff / (1. + Noffregions)
    return Poisson_logL(Non, Noff, mu_gam, mu_bg, Noffregions)

def Poisson_logL_Noff0(Non, Noff, mu_gam, Noffregions):
    fAlpha = 1/Noffregions
    mu_bg = fAlpha * Non / (1 + fAlpha) - mu_gam
    if mu_bg < 0.:
        mu_bg = 0
    return Poisson_logL(Non, Noff, mu_gam, mu_bg, Noffregions)

def Poisson_logL_else(Non, Noff, mu_gam, Noffregions):
    mu_bg = mu_BG(mu_gam, Non, Noff, Noffregions)
    return Poisson_logL(Non, Noff, mu_gam, mu_bg, Noffregions)

def Gauss_logL(Non, Noff, mu_gam, Noffregions):
    diff = Non - Noff/Noffregions - mu_gam
    delta_diff = np.sqrt(Non + Noff/Noffregions) 
    return np.square(diff)/np.square(delta_diff)

def find_z(possible_z, source_z):
    idx = np.argmin(np.abs(possible_z - source_z))
    return possible_z[idx]

#Concave EBL method functions:

def find_tg_crossing(energy, logE, logemtau):
    """
    Find where the tangent of the curve of logemtau at energy cuts the loemtau curve

    Parameters 
    ----------
    energy : float
        Energy where we want to compute the concave EBL in TeV
    logE : array_like
        Logarithm of the energy where we want to compute the concave EBL in TeV
    logemtau : array_like
        Logarithm of the exponential of the optical depth
    """
    tck = interpolate.splrep(logE, logemtau)
    x0 = np.log10(energy)
    y0 = interpolate.splev(x0,tck)
    dydx = interpolate.splev(x0,tck,der=1)
    f = interpolate.splev(logE,tck)
    g = dydx*(logE-x0) + y0
    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
    return idx

def EBL_concave(Energy, z, EBL_Model):
    """
    Uses an EBL model optical depth at a certain redshift to make an EBL optical depth which has no inflection point (concave).

    Parameters
    ----------
    Energy : array_like
        Energy where we want to compute the concave EBL in TeV
    z : float
        Redshift of the source
    EBL_Model : str
        EBL model to use for the optical depth (see ebltable documentation for more info)  
    """
    tau1 =  OptDepth.readmodel(model=EBL_Model)
    tau_sim = tau1.opt_depth(z, Energy)

    logE = np.log10(Energy)
    logemtau = (-tau_sim)/np.log(10)

    tck = interpolate.splrep(logE, logemtau)

    testsEs = np.geomspace(0.1, 10, 1000)
    for i, E in enumerate(testsEs):
        cross = find_tg_crossing(E, logE, logemtau)
        if (len(cross) > 0):
            if ((cross[1] - cross[0] > 2) & (Energy[cross[0]] > 1.)):
                break

    Etg1 = testsEs[i-1]
    Etg2 = 10**logE[int(np.around(np.mean((cross[0], cross[1]))))]
    tck = interpolate.splrep(logE,logemtau)
    x0 = np.log10(Etg1)
    x1 = logE[int(np.around(np.mean((cross[0], cross[1]))))]
    y0 = interpolate.splev(x0,tck)
    y1 = interpolate.splev(x1,tck)
    dydx = (y1 - y0) / (x1 - x0)   #interpolate.splev(x0,tck,der=1)
    tngnt = lambda x: dydx*x + (y0-dydx*x0)

    concavetau = np.exp(-tau_sim)

    tangentvalues = (10**tngnt(logE[np.where((Energy<Etg2) & (Energy>Etg1))]))
    concavetau[np.where((Energy<Etg2) & (Energy>Etg1))] = tangentvalues

    concavetau = -np.log(concavetau)
    return concavetau