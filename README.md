Expected output of the code si shown [here](z_boson_plot_result.png)

# FILTERING:
This code reads in an arbitrary number of datafiles, filters the data to
exclude non-numerical lines/ elements including nans and infs. the data is then
filtered based on physical motivation(i.e only nonzero uncertainties etc), then
the average around each datapoint is found, if the datapoint is too different
from the average (by default 3 times bigger) it is classed as an outlier. the
function is then fit to the average of the datapoints and the standard
deviations of each datapoint is used to exclude datapoints 3 or more standard
deviations away.

# FITTING:
the code can find an initial guess based on FWHM and maximising ordinate value,
or the code uses the provided initial guess (depends on truth value of AUTO_INITIAL_GUESS)
the validated data is then used to fit the equation and find the minimised
chi squared parameters, the values found are:
    width of z boson
    mass of z boson
    minimised chi squared value
    reduced chi squared

# CALCULATION:
the minimised plot and values are used to calculate:
    maximum cross section
    maximising energy
    maximum rate of event occurence (instantanous luminosity given)
    lifetime of the Z boson
    the full width half maximum of the curve

The equation to fit is:
    cross_section = $(12*pi / m^2)*(E^2 / ((E^2 - m^2)+(mw_z)^2)) * (w_ee)^2$
    where:
        cross_section = cross section of the interaction
            in natual units [GeV^-2]
        E = centre of mass energy of in [GeV]
        m = mass of the Z_0 boson in [Gev/c^2]
        w_z =  width of the Z_0 boson in [GeV]
        w_ee = parial width for Z_0 boson -> electron positron decay in [GeV]
    the equation is in natural units

The rate of event occuring can be given by:
    R = cross_section * L
    where:
        R - rate of pair production event in [seconds^-1]
        L - instantanous luminosity of colliding beams in [area^-1 second^-1]
with values for m and w_z, a lifetime of Z_0 boson can be calculated by:
    t_w = h_bar^2/w_z
    where:
        t_w - lifetime of Z_0 boson in seconds
        L - insta
the data is plotted against the calculated model, and a contour plot showing
the chi squared as a function of parameter space is made.
provided a beam luminosity the rate R a peak reaction rate is calculated using
the highest value of cross section according to the function:
    R = cross_section * beam luminsoity
