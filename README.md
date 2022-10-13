# EBL_fit_MC
Fits to a simulated absorved blazar spectrum in order to validate EBL models.

The usefull code is in "EBL_fit_MC.ipynb". There are other files related to fold or fitebl.

In the code there are some options to choose, Telescope (MAGIC will be added soon), Fit function (PWL, LP, freeLP and MBPWL at the moment) (if you choose the MBPWL you can also choose the position of the first and last knots (the other knots are logaritmically distributed between those two values), simulated spectra function ("PWL" at the moment and "LP" soon), source flux, Observation time and number of iterations. There is also the option to use the energy migration or not.

The code first simulates an EBL absorbed spectrum (using Dominguez et al. 2011 EBL model). After that it simulates how the selected telescope would see it (Effective Area, Energy resolution, Background and Angular resolution). After that, the program looks for the best fit for every apha value (deabsorption coefficient) and plots it in a chi^2 vs alpha plot. 

In order to run the code with condor (most updated simulation), introduce the desired configuration with the config creator and then run EBL_mult2.sub. (if you want to run a number of realizations different to 50 or 1000, you will have to use txtitercreator.ipynb and change the file in EBL_mult2.sub.) You also need to change the path in EBL_mult2.sh.
