import numpy as np
import scipy.interpolate as interpolate
from scipy.integrate import quad
from scipy.stats import poisson

def chisq(obs, exp, error):
    return np.sum(np.square(obs - exp) / np.square(error))
    
def Gauss(E, A, mu, sigma):
    return A * (np.exp(-1/2 * np.square((E - mu) / sigma)) / (sigma * np.sqrt(2*np.pi)))

def Gauss_int(A, mu, sigma, Em, Ep):
    return quad(Gauss, Em, Ep, args=(A, mu, sigma)) #maybie some mistake here because of log scale


def log_interp1d(E_before, y_before, E_after):
    interp_func = interpolate.interp1d(np.log10(E_before), np.log10(y_before), bounds_error=False, fill_value = "extrapolate", kind='linear')
    interpolated = np.power(10 ,interp_func(np.log10(E_after)))
    return interpolated

def log_interp1d2(xx, yy):
    logx = np.log10(xx)
    logy = np.log10(yy)
    interp = interpolate.interp1d(logx, logy, fill_value='extrapolate', kind='slinear')
    log_interp = lambda zz: np.power(10.0, interp(np.log10(zz)))
    return log_interp


def normal_interp1d(E_before, y_before, E_after):
    interp_func = interpolate.interp1d(E_before, y_before, bounds_error=False, fill_value = "extrapolate", kind='linear')
    interpolated = interp_func(E_after)
    return interpolated


def SED_gen(rng_num, bckgmu, mu_vec, Effa, Ebinsw, Observation_time, E, Nwobbles):
    my_generator = np.random.default_rng(rng_num)
    Simbckg1 = my_generator.poisson(bckgmu)
    # Simbckg1 = Simbckg1.astype(float)
    # for i in range(len(Simbckg1)):
    #     if Simbckg1[i] == 0:
    #         Simbckg1[i] = bckgmu[i]
    Simbckg1_u = np.sqrt(Simbckg1)
    Simbckg5 = my_generator.poisson(Nwobbles*bckgmu)/Nwobbles
    # Simbckg5 = Simbckg5.astype(float)
    # for i in range(len(Simbckg5)):
    #     if Simbckg5[i] == 0:
    #         Simbckg5[i] = bckgmu[i]
    Simbckg5_u = np.sqrt(Simbckg5)

    N = my_generator.poisson(mu_vec)
    # N[N==0] = 1
    N_u = np.sqrt(N)

    NpB = N + Simbckg1 - Simbckg5
    NpB_u = np.sqrt(N_u**2 + Simbckg1_u**2 + Simbckg5_u**2)
    NpB_u[NpB_u == 0] = 1


    dNdE_b = NpB / Effa / Ebinsw / Observation_time
    dNdE_b_u = NpB_u / Effa / Ebinsw / Observation_time

    SED = np.square(E) * dNdE_b
    SED_u = np.square(E) * dNdE_b_u
    return SED, SED_u, dNdE_b, dNdE_b_u

def fit_func_select(fit_func_name, knots = 3, Efirst = 0.2 , Elast = 1.12):
    if fit_func_name == "MBPWL":
        def fit_func(xdata, params):
            if knots < 2 or knots > 30: #change this when adding more number of knots
                raise Exception('knots have to be larger or equal than 3 and smaller than 30')
            else:
                polw = np.zeros(len(xdata))
                Ebr = np.geomspace(Efirst, Elast, knots)
                gamma = np.zeros(knots+1)
                phi = np.zeros(knots+1)
                phi_0 = params[0] #len(sqrtdelta_lam) = len(lam)-1 = len(phi)-1
                gamma0 = params[1]
                sqrtdelta_gamma = params[2:knots+2]
                delta_gamma = np.square(sqrtdelta_gamma)
                gamma[0] = gamma0
                phi[0] = phi_0
                for i in range(knots):
                    gamma[i+1] = gamma[i] + delta_gamma[i]
                    phi[i+1] = phi[i] * Ebr[i] ** delta_gamma[i]
                for i in range(len(xdata)):
                    for j in range(knots):
                        if xdata[i]<Ebr[0]:
                            polw[i] = phi[0] * (xdata[i]/0.25) ** (-gamma[0])
                        elif Ebr[-1] < xdata[i]:
                            polw[i] = phi[-1] * (xdata[i]/0.25) ** (-gamma[-1])
                        elif Ebr[j] <= xdata[i] < Ebr[j+1]:
                            polw[i] = phi[j+1] * (xdata[i]/0.25) ** (-gamma[j+1])
            return polw
        return(fit_func)

    elif fit_func_name == "PWL":
        def fit_func(xdata, params):
            phi = params[0]
            gamma = params [1]
            PLW = phi / ((xdata/0.25) ** gamma)
            return PLW
        return(fit_func)

    elif fit_func_name == "LP":
        def fit_func(xdata, params):
            phi0 = params[0]
            alpha = params[1]
            beta = params[2]
            #Enorm = 1TeV #if it is 1 TeV no need ot include it (if it is different, need to add it to te LP function)
            LP = phi0 * np.power((xdata/0.25), (-alpha - beta * beta * np.log(xdata/0.25)))
            return LP
        return(fit_func)

    elif fit_func_name == "freeLP":
        def fit_func(xdata, params):
            phi0 = params[0]
            alpha = params[1]
            beta = params[2]
            #Enorm = 1TeV #if it is 1 TeV no need ot include it (if it is different, need to add it to te LP function)
            LP = phi0 * np.power((xdata/0.25), (-alpha - beta * np.log(xdata/0.25)))
            return LP
        return(fit_func)

    else:
        raise Exception('The function "{func}" has not been implemented.'.format(func = fit_func_name))

def alphas_creation(initial_guess_pos, first_bin, last_bin, step):
    alphas1 = np.arange(initial_guess_pos + step, last_bin + step, step)
    alphas2 = np.arange(initial_guess_pos, first_bin - step, -step)
    alphas = np.append(alphas1, alphas2)
    return alphas    

def ig_mat_create(fit_func_name, alphas, knots):
    if fit_func_name == "MBPWL":
        initial_guess_mat = np.zeros((len(alphas)+1, knots+2))
    elif fit_func_name == "PWL":
        initial_guess_mat = np.zeros((len(alphas)+1, 2))
    elif fit_func_name == "LP":
        initial_guess_mat = np.zeros((len(alphas)+1, 3))
    return initial_guess_mat

def SED_alpha(alpha, dNdE_b, dNdE_b_u, tau, E_final):
    dNdE2 = dNdE_b * np.exp(alpha*tau)
    dNdE2_u = dNdE_b_u * np.exp(alpha*tau)
    SED = np.square(E_final) * dNdE2
    SED_u = np.square(E_final) * dNdE2_u
    return SED, SED_u

def sigma_intervals(sigma, chis_new, step, interpx):
    sigma_inter = np.where(chis_new <= sigma**2 + np.min(chis_new))
    upper_bound = step/5 * np.max(sigma_inter)
    lower_bound = step/5 * np.min(sigma_inter)
    if sigma == 1:
        print("The minimum is at alpha = ", interpx[np.argmin(chis_new)], " + ", upper_bound-interpx[np.argmin(chis_new)], " - ", interpx[np.argmin(chis_new)] - lower_bound)
    else:
           print("The {sigma} sigma interval is at alpha = ".format(sigma = sigma), interpx[np.argmin(chis_new)], " + ", upper_bound-interpx[np.argmin(chis_new)], " - ", interpx[np.argmin(chis_new)] - lower_bound)
 
def ordering_sigma(alphas, first_bin, last_bin, step, chisqs):
    order = np.argsort(alphas)
    interpx = np.arange(first_bin, last_bin, step/5)
    alphas_reord = np.take_along_axis(alphas, order, axis=0)
    chisqs_reord = np.take_along_axis(np.array(chisqs), order, axis=0)
    f1 = interpolate.interp1d(alphas_reord, chisqs_reord, kind='linear')
    chis_new = f1(interpx)
    sigma_intervals(1, chis_new, step, interpx)
    sigma_intervals(2, chis_new, step, interpx)
    sigma_intervals(3, chis_new, step, interpx)

def on_off_rnd(rng_num, bckgmu, mu_vec, Nwobbles):
    my_generator = np.random.default_rng(rng_num)
    Simbckg1 = my_generator.poisson(bckgmu)
    Simbckg5 = my_generator.poisson(Nwobbles*bckgmu)/Nwobbles
    N = my_generator.poisson(mu_vec)
    # N[N==0] = 1

    ON = N + Simbckg1
    OFF = Simbckg5

    return ON, OFF

def SED_gen_nobckg(rng_num, mu_vec, Effa, Ebinsw, Observation_time, E):
    my_generator = np.random.default_rng(rng_num)
    NpB = my_generator.poisson(mu_vec)
    NpB_u = np.sqrt(NpB)
    NpB_u[NpB_u == 0] = 1

    dNdE_b = NpB / Effa / Ebinsw / Observation_time
    dNdE_b_u = NpB_u / Effa / Ebinsw / Observation_time

    SED = np.square(E) * dNdE_b
    SED_u = np.square(E) * dNdE_b_u
    return SED, SED_u, dNdE_b, dNdE_b_u

def mu_BG(mu_g, Non, Noff): #check if Nwobbles has to be applied here
    mubg = ((-6 * mu_g) + Non + Noff + np.sqrt(np.square((6 * mu_g) - Non - Noff) + (24 * Noff * mu_g)))/12
    return mubg

# def N_rnd(rng_num, mu):
#     my_generator = np.random.default_rng(rng_num)
#     N = my_generator.poisson(mu)
#     return N

def FF_Likelihood(Non, Noff, mu_gamma, mu_bg, Nwobbles):
    mu_on = mu_gamma + mu_bg
    mu_off = mu_bg * Nwobbles
    L = np.sum(poisson.pmf(k = Non, mu = mu_on) * poisson.pmf(k = Noff, mu = mu_off))
    return L

def dNdE_to_mu(dNdEa, Effa_reb, Ebinsw, Observation_time, Ebins, Eres_reb2, E_EBL):
    mu_vec = dNdEa * Effa_reb * Ebinsw * Observation_time

    mu_vec_reco = np.zeros(len(mu_vec))
    mu_vec_i = np.zeros(len(mu_vec))

    for i in range(len(mu_vec)):
        for j in range(len(mu_vec)):
            A = mu_vec[i]
            Em = Ebins[j]
            Ep = Ebins[j+1]
            sigma = Eres_reb2[i]
            mu = E_EBL[i]

            mu_vec_i[j] = Gauss_int(A, mu, sigma, Em, Ep)[0]
        mu_vec_reco = mu_vec_reco + mu_vec_i
     
    return mu_vec_reco

def dNdE_to_mu_MAGIC(dNdEa, Ebinw, migmatval, Eest):
    mu_vec = dNdEa * Ebinw
    mu_vec_reco = np.zeros(len(Eest))
    for i in range(len(Eest)):
        mu_vec_reco[i] = np.sum(mu_vec * migmatval[:,i])
    return mu_vec_reco


def Poisson_logL(Non, Noff, mu_gam, mu_bg, Nwobbles):
    # print(Non, Noff, mu_gam, mu_bg)
    logL = np.log(poisson.pmf(Non, mu_gam + mu_bg) * poisson.pmf(Noff, Nwobbles * mu_bg)) #change this to sum of logs.(and maybie add gaussian)
    logLmax = np.log(poisson.pmf(Non, Non) * poisson.pmf(Noff, Noff))
    return -2 * (logL - logLmax)

def Poisson_logL_Non0(Non, Noff, mu_gam, Nwobbles):
    mu_bg = Noff / (1.+Nwobbles)
    return Poisson_logL(Non, Noff, mu_gam, mu_bg, Nwobbles)

def Poisson_logL_Noff0(Non, Noff, mu_gam, Nwobbles):
    fAlpha = 1/Nwobbles
    mu_bg = fAlpha * Non / (1 + fAlpha) - mu_gam
    for i in range(len(mu_bg)):
        if mu_bg[i] < 0.:
            mu_bg[i] = 0
    return Poisson_logL(Non, Noff, mu_gam, mu_bg, Nwobbles)

def Gauss_logL(Non, Noff, mu_gam, Nwobbles):
    diff = Non - Noff/Nwobbles - mu_gam 
    delta_diff = np.sqrt(Non + Noff/Nwobbles) 
    return np.square(diff)/np.square(delta_diff)