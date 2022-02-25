# EBL_fit_MC
Fits to a simulated absorved blazar spectrum in order to validate an EBL models.

The usefull code is in "EBL_fit_MC.ipynb". There are other files related to fold or fitebl.

In the code there are some options to choose, Telescope (MAGIC will be added soon), Fit function (PWL, LP, freeLP and MBPWL at the moment), simulated spectra function ("PWL" at the moment and "LP" soon), source flux, Observation time and number of iterations. There is also the option to use the energy migration or not.

The code first simulates an EBL absorbed spectrum (using Dominguez et al. 2011 EBL model). After that it simulates how the selected telescope would see it (Effective Area, Energy resolution, Background and Angular resolution). After that, the program looks for the best fit for every apha value (deabsorption coefficient) and plots it in a chi^2 vs alpha plot. 
