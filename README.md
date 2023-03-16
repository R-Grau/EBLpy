# EBL_fit_MC
Fits to a simulated absorved blazar spectrum in order to validate EBL models.

The code to run is "EBL_MC_final.py" which is the Toy MC simulation. It has to be run multiple times with different inputs (e.g. "python EBL_MC_final.py 0", "python EBL_MC_final.py 1", ...). Change the "pathstring" variable in "EBL_MC_final.py" and "EBL_MC_final_functions.py" to match the path where you have all the needed files and where you want to store the output.
In order to read the output use "EBL_fit_MC_viewer.ipynb" section called "Multiple files with data". 

Use "EBL_MC_final_config_creator.ipynb" to generate or modify the needed configuration files depending on your needs.
At this moment the CTAN-alpha option in Telescopes has not been implemented yet. Other important options are the intrinsic spectrum shape and the fit unction.

The code first simulates an EBL absorbed spectrum (using Dominguez et al. 2011 EBL model). After that it simulates how the selected telescope would see it (Effective Area, Energy resolution, Background and Angular resolution). After that, the program looks for the best fit for every apha value (deabsorption coefficient) and plots it in a -2logLikelihood vs alpha plot. 

In order to run the code with condor, generate the needed config files and modify "EBL_MC_final.sub" (if you want to run a number of realizations different to 50 or 1000, you will have to use txtitercreator.ipynb and change the file in "EBL_MC_final.sub".) You also need to change the path in "EBL_MC_final.sh".
