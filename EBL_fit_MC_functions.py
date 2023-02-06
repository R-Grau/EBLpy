import numpy as np
import scipy.interpolate as interpolate
from scipy.integrate import quad
from scipy.stats import poisson
import pandas as pd

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

def tau_interp(E_after, z_after, EBL_Model, kind_of_interp = "linear"):
    if EBL_Model == "Dominguez":
        possible_z = np.array([0.01, 0.02526316, 0.04052632, 0.05578947, 0.07105263, 0.08631579, 0.10157895, 0.11684211, 0.13210526, 0.14736842, 0.16263158, 0.17789474, 0.19315789, 0.20842105, 0.22368421, 0.23894737, 0.25421053, 0.26947368, 0.28473684, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
        file = np.loadtxt('/data/magic/users-ifae/rgrau/EBL-splines/tau_dominguez11.out')#np.loadtxt('/home/rgrau/Desktop/EBL-splines/tau_dominguez11.out')
        pdfile = pd.DataFrame(file)
        pdfile = pdfile.rename(columns={ 0 : 'E [TeV]', 1: 'tau z=0.01', 2: 'tau z=0.02526316', 3: 'tau z=0.04052632', 4: 'tau z=0.05578947', 5: 'tau z=0.07105263', 6: 'tau z=0.08631579', 7: 'tau z=0.10157895', 8: 'tau z=0.11684211', 9: 'tau z=0.13210526', 10: 'tau z=0.14736842', 11: 'tau z=0.16263158', 12: 'tau z=0.17789474', 13: 'tau z=0.19315789', 14: 'tau z=0.20842105', 15: 'tau z=0.22368421', 16: 'tau z=0.23894737', 17: 'tau z=0.25421053', 18: 'tau z=0.26947368', 19: 'tau z=0.28473684', 20: 'tau z=0.3' , 21: 'tau z=0.35', 22: 'tau z=0.4' , 23: 'tau z=0.45', 24: 'tau z=0.5', 25: 'tau z=0.55', 26: 'tau z=0.6', 27: 'tau z=0.65', 28: 'tau z=0.7' , 29: 'tau z=0.75'})
        E_before = pdfile['E [TeV]'].to_numpy() #energy bins
        tau_matrix = np.zeros([len(possible_z), len(E_before)])
        for i in range(len(possible_z)):
            tau_matrix[i] = pdfile['tau z={0}'.format(possible_z[i])].to_numpy() #tau bins
    else:
        raise Exception('The EBL model "{func}" has not been implemented.'.format(func = EBL_Model))        

    if kind_of_interp == "linear":
        interpolation = interpolate.interp2d(E_before, possible_z, tau_matrix)
        tau_new = interpolation(E_after, z_after)
    elif kind_of_interp == "log":
        log_interpolation = interpolate.interp2d(np.log10(E_before), np.log10(possible_z), np.log10(tau_matrix))
        tau_new = np.power(10, log_interpolation(np.log10(E_after), np.log10(z_after)))

    return(tau_new)

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
                phi[1] = phi[0] * (Eknot/0.25) ** delta_gamma[0]
                for i in range(len(xdata)):
                    if xdata[i] < Eknot:
                        polw[i] = phi[0] * (xdata[i]/0.25) ** (-gamma[0])
                    elif xdata[i] >= Eknot:
                        polw[i] = phi[1] * (xdata[i]/0.25) ** (-gamma[1])
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
                    phi[i+1] = phi[i] * (Ebr[i]/0.25) ** delta_gamma[i]
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
            freeLP = phi0 * np.power((xdata/0.25), (-alpha - beta * abs(beta) * np.log(xdata/0.25)))
            return freeLP
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
    elif fit_func_name == "LP" or fit_func_name == "freeLP":
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
    Simbckg_wob = my_generator.poisson(Nwobbles*bckgmu)/Nwobbles
    N = my_generator.poisson(mu_vec)

    ON = N + Simbckg1
    OFF = Simbckg_wob

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

def mu_BG(mu_g, Non, Noff, Nwobbles):
    mubg = ((-(Nwobbles+1) * mu_g) + Non + Noff + np.sqrt(np.square(((Nwobbles+1) * mu_g) - Non - Noff) + (4*(Nwobbles+1) * Noff * mu_g)))/(2*(Nwobbles+1))
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


def Poisson_logL_IRF(Non, Noff, mu_gam, delta_mu_gam, mu_bg, Nwobbles): #expectedgammas = mu_gam, bckg = Noff/Nwobbles, observed = Non
    logL = np.log(poisson.pmf(Non, mu_gam + mu_bg)) + np.log(poisson.pmf(Noff, Nwobbles * mu_bg))
    logLmax = np.log(poisson.pmf(Non, Non) * poisson.pmf(Noff, Noff))
    return -2 * (logL - logLmax)

def Poisson_logL_Non0_IRF(Non, Noff, mu_gam, delta_mu_gam, Nwobbles): #canviat per IRF
    mu_bg = Noff / (1. + Nwobbles)
    mu_gam2 = -np.square(delta_mu_gam) + mu_gam
    for i in range(len(mu_gam2)): #FIXME ?
        if mu_gam2[i] < 0.:
            mu_gam2[i] = 0
    return Poisson_logL_IRF(Non, Noff, mu_gam2, delta_mu_gam, mu_bg, Nwobbles)

def Poisson_logL_Noff0_IRF(Non, Noff, mu_gam, delta_mu_gam, Nwobbles):
    fAlpha = 1/Nwobbles
    mu_bg = fAlpha * Non / (1 + fAlpha) - mu_gam -np.square(delta_mu_gam)/fAlpha
    for i in range(len(mu_bg)):
        if mu_bg[i] < 0.:
            mu_bg[i] = 0
            a = 1.
            b = -mu_gam + np.square(delta_mu_gam)
            c = -Non + np.square(delta_mu_gam)
            mu_gam2 = (-b + np.sqrt(np.square(b) - 4 * a * c)) / (2. * a)
        else:
            mu_gam2 = mu_gam + np.square(delta_mu_gam) / fAlpha
    return Poisson_logL_IRF(Non, Noff, mu_gam2, delta_mu_gam, mu_bg, Nwobbles)

def Gauss_logL_IRF(Non, Noff, mu_gam, delta_mu_gam, Nwobbles): #canviat per IRF
    Noff_n = Noff/Nwobbles #Noff_n_unc = np.sqrt(Noff_n) but as we have to ^2 later we just don't define it. (same for Non_u)
    diff = Non - Noff_n - mu_gam #was Non - Noff/Nwobbles - mu_gam
    delta_exp = np.sqrt(np.square(delta_mu_gam) + Noff_n)
    delta_diff = np.sqrt(Non + np.square(delta_exp)) 
    return np.square(diff)/np.square(delta_diff)

###################remoove this later
def Poisson_logL(Non, Noff, mu_gam, mu_bg, Nwobbles):
    # print(Non, Noff, mu_gam, mu_bg)
    logL = np.log(poisson.pmf(Non, mu_gam + mu_bg)) + np.log10(poisson.pmf(Noff, Nwobbles * mu_bg))
    logLmax = np.log(poisson.pmf(Non, Non) * poisson.pmf(Noff, Noff))
    return -2 * (logL - logLmax)

def Poisson_logL_Non0(Non, Noff, mu_gam, Nwobbles):
    mu_bg = Noff / (1. + Nwobbles)
    return Poisson_logL(Non, Noff, mu_gam, mu_bg, Nwobbles)

def Poisson_logL_Noff0(Non, Noff, mu_gam, Nwobbles):
    fAlpha = 1/Nwobbles
    mu_bg = fAlpha * Non / (1 + fAlpha) - mu_gam
    for i in range(len(mu_bg)):
        if mu_bg[i] < 0.:
            mu_bg[i] = 0
    return Poisson_logL(Non, Noff, mu_gam, mu_bg, Nwobbles)

def Gauss_logL(Non, Noff, mu_gam, Nwobbles):
    diff = Non - Noff/Nwobbles - mu_gam #was Non - Noff/Nwobbles - mu_gam
    delta_diff = np.sqrt(Non + Noff/Nwobbles) 
    return np.square(diff)/np.square(delta_diff)
#########################################################

def find_z(possible_z, source_z):
    idx = np.argmin(np.abs(possible_z - source_z))
    return possible_z[idx]