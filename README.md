Basic python script for filtering data<br>
Check the src folder for code and data used.<br>
Expected output of the code is ![](src/z_boson_plot_result.png)


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
chi squared parameters, the values found are:<br>
    &emsp;width of z boson <br>
    &emsp;mass of z boson<br>
    &emsp;minimised chi squared value<br>
    &emsp;reduced chi squared<br>

# CALCULATION:
the minimised plot and values are used to calculate:<br>
    &emsp;maximum cross section<br>
    &emsp;maximising energy<br>
    &emsp;maximum rate of event occurence (instantanous luminosity given)<br>
    &emsp;lifetime of the Z boson<br>
    &emsp;the full width half maximum of the curve<br>

The equation to fit is:<br>
$$\text{Cross section} = \large \dfrac{12\pi}{m_z^2}\dfrac{E^2}{ (E^2 - m_z^2)^2+(m_zw_z)^2}w_{ee}^2$$<br>
    where:<br>
        &emsp;cross_section = cross section of the interaction in natural units [GeV^-2]<br>
        &emsp;E = centre of mass energy of in [GeV]<br>
        &emsp;m = mass of the Z_0 boson in [Gev/c^2]<br>
        &emsp;w_z =  width of the Z_0 boson in [GeV]<br>
        &emsp;w_ee = parial width for Z_0 boson -> electron positron decay in [GeV]<br>
    the equation is in natural units<br>

The rate of event occuring can be given by:<br>
    &emsp;R = cross_section * L<br>
    where:<br>
        &emsp;R - rate of pair production event in [seconds^-1]<br>
        &emsp;L - instantanous luminosity of colliding beams in [area^-1 second^-1]<br>
with values for m and w_z, a lifetime of Z_0 boson can be calculated by:<br>
    &emsp;t_w = h_bar^2/w_z<br>
    where:<br>
        &emsp;t_w - lifetime of Z_0 boson in seconds<br>
the data is plotted against the calculated model, and a contour plot showing
the chi squared as a function of parameter space is made.
provided a beam luminosity the rate R a peak reaction rate is calculated using
the highest value of cross section according to the function:<br>
    &emsp;R = cross_section * beam luminsoity
