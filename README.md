# EBL-splines
Spline fit to a simulated absorved blazar spectrum in order to validate an EBL models.

At this moment we have different versions:

One with Gaussian noise.

Another one simulates real observations by adding Poisson distribution to the data.

The third one simulates real observations with Poisson distribution into the data and then simulates the background and backgorund subtraction process to the data. This process adds extra uncertainty to the data and of course makes it fit worse with the model.

We then have different jupyter notebooks:

-The first one ("Multiple_Broken_power_law.ipynb") has a simulated spectra, absorved by an EBL model. It also includes the background, Effective Area and Angular resolution of CTA-N in alpha configuration.
-The second one ("Multiple_Broken_power_law2.ipynb") is like the previous one but includes the Energy resolution of CTA individualizing the photons.
-The third one ("Multiple_Broken_power_law3.ipynb") is the same but uses a different way to include the energy resolution which allows us to get a correct uncertainty for the final data. It also includes comments in order to understand and play with the code.

-The forth and fifth ("Multiple_Broken_power_law_multiple.ipynb" and "Multiple_Broken_power_law_multiple2.ipynb") are like the second and third but, instead of computing the case of one observation, they allow the iteration of the process.
