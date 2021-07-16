# EBL-splines
Spline fit to a simulated absorved blazar spectrum in order to validate an EBL model.

At this moment we have 3 versions:

One with Gaussian noise.

Another one simulates real observations by adding Poisson distribution to the data.

The third one simulates real observations with Poisson distribution into the data and then simulates the background and backgorund subtraction process to the data. This process adds extra uncertainty to the data and of course makes it fit worse with the model.

We also have a jupyter notebook for a broken power law to fit the data. In the future we will implement a multiple broken power law to fit the data depending on the number of knots.
