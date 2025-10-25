# -*- coding: utf-8 -*-
"""
Title:
    PHYS20161 – 2nd Assignment: Z_0 boson

FILTERING:
This code reads in an arbitrary number of datafiles, filters the data to
exclude non-numerical lines/ elements including nans and infs. the data is then
filtered based on physical motivation(i.e only nonzero uncertainties etc), then
the average around each datapoint is found, if the datapoint is too different
from the average (by default 3 times bigger) it is classed as an outlier. the
function is then fit to the average of the datapoints and the standard
deviations of each datapoint is used to exclude datapoints 3 or more standard
deviations away.

FITTING:
the code can find an initial guess based on FWHM and maximising ordinate value,
or the code uses the provided initial guess (depends on truth value of
                                             AUTO_INITIAL_GUESS)
the validated data is then used to fit the equation and find the minimised
chi squared parameters, the values found are:
    width of z boson
    mass of z boson
    minimised chi squared value
    reduced chi squared

CALCULATION:
the minimised plot and values are used to calculate:
    maximum cross section
    maximising energy
    maximum rate of event occurence (instantanous luminosity given)
    lifetime of the Z boson
    the full width half maximum of the curve

The equation to fit is:
    cross_section = (12*pi / m^2)*(E^2 / ((E^2 - m^2)+(mw_z)^2)) * (w_ee)^2
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

Note for Windows users:
    bug may occur where a stray datapoint is added at ~250 GeV , the calculated
    values do not change but the plot does not format well, retry in a new
    console or running the code again may help - no problems occur on mac

Last Updated: 21/12/23 (editing after assignment was due)
author ID: d17397nr

"""
import time as timer
import numpy as np
import scipy.constants as pc
import matplotlib.pyplot as plt

# file constants
FILE1 = 'z_boson_data_1.csv'
FILE2 = 'z_boson_data_2.csv'
FILES = [FILE1, FILE2]  # increase sample size by adding to this list

PRINT_ERROR_LINES = True  # print which datapoints were removed
DELIMITER = ','  # all files to have same delimiter
FORCE_POSITIVE_UNCERTAINTY = True  # converts data array 3rd column to positive

# value constants
EE_PARTIAL_WIDTH = 83.91 * 10**-3  # [GeV]
BEAM_LUMINOSITY = 1e+34  # [cm^-2 s^-1]

# plotting constants
SAVE_FIGURE = False
FIGURE_NAME = 'z_boson_plot_result.png'  # extension must be included
GRAPH_STYLE = 'ggplot'
GRAPH_COLOR_DATAPOINT = 'blue'
GRAPH_COLOR_FUNCTION = 'red'

CONTOUR_STYLE = 'plasma'
CONTOUR_COLOR = 'black'

PLOT_FWHM = True
FWHM_COLOR = 'black'
FWHM_ALPHA = 0.7
FWHM_LINESTYLE = 'dashed'

PLOT_MAXIMISING_ENERGY = True
ENERGY_LINE_COLOR = 'black'
ENERGY_LINE_ALPHA = 0.7
ENERGY_LINE_LINESTYLE = '-.'

# minimisation constants
AUTO_INITAL_GUESS = True  # if true finds own intial guess
MINIMISATION_INITIAL_GUESS = [90, 3]  # mass[GeV/c**2], width[GeV]

MINIMISATION_STEP_SIZE = 0.01
MINIMISATION_TOLERANCE = 0.000001
MINIMISATION_MAX_COUNTER = 100000

# data validation functions


def read_data(file_name, delimiter=DELIMITER,
              number_columns=3, print_error_lines=PRINT_ERROR_LINES):
    """
    function takes in a filename, checks for existence, opens it for reading,
    iterates over every line and appends the line to an array if the line:
        has the required number of columns
        every element can be floated
        is not nan or inf

    Parameters
    ----------
    file_name : string
        name of file with extension included.
    delimiter : string, optional
        the character seperating data in the file. The default is DELIMITER.
    number_columns : int, optional
        number of columns the datafile is expexted to have. The default is 3.
    print_error_lines : Bool, optional
        prints what lines are invalid. The default is PRINT_ERROR_LINES.

    Raises
    ------
    SystemExit
        if the file cannot be found, the code will not run.

    Returns
    -------
    data_array : numpy array
        array of validated data, with {number_columns} number of columns.

    """
    try:
        file = open(file_name, 'r')
    except FileNotFoundError as e:
        print(f"The file '{file_name}' cannot be found, check name or "
              "directory")
        print(e)
        raise SystemExit() from e
    data = np.zeros((1, number_columns))
    counter = 0
    for line in file:
        counter += 1
        if line[0] in ['#', '%']:
            continue
        line = line.strip('\n')
        line_split = line.split(delimiter)
        if len(line_split) != number_columns:
            if print_error_lines:
                print(f'file line {counter} did not have {number_columns} '
                      f'columns ({file_name})')
            continue
        line_validated = []
        try:
            for element in enumerate(line_split):
                float(element[1])
                if np.isnan(float(element[1])):
                    if print_error_lines:
                        print(f'line {counter} element '
                              f'{element[0]+1} is nan {file_name}')
                    continue
                if np.isinf(float(element[1])):
                    if print_error_lines:
                        print(f'line {counter} element '
                              f'{element[0]+1} is inf {file_name}')
                    continue
                line_validated.append(float(element[1]))
            data = np.vstack((data, line_validated))
        except ValueError:
            if print_error_lines:
                print(f"line {counter} was non numerical {file_name}")
            continue
    file.close()
    data_array = np.delete(data, 0, 0)
    return data_array


def combine_array(*array):
    """
    takes a variable amount of arrays and vertically stacks them,
    the number of columns is decided by the length of the first arrays first
    line

    Parameters
    ----------
    *array : array
        the variable number of arrays to stack.

    Raises
    ------
    SystemExit
        if the number of columns is not constant between arrays.

    Returns
    -------
    sorted_data : numpy array
        one list consiting of combined input arrays and sorted by ascending
        order of first element.

    """
    number_columns = len(array[0][0])
    combined_array = np.zeros((1, number_columns))
    for array_to_stack in array:
        if len(array_to_stack[0]) != number_columns:
            print('the number of columns between arrays is not constant'
                  ' so they cannot be stacked - check the number of columns')
            raise SystemExit()
        combined_array = np.vstack((combined_array, array_to_stack))
    combined_array = np.delete(combined_array, 0, 0)
    sorted_data = np.array(sorted(combined_array, key=lambda x: x[0]))
    return sorted_data


def read_files_and_combine(filename_array, columns):
    """
    for a list of filenames, opens the file, reads the data in each file and
    combines the data into one large array

    Parameters
    ----------
    filename_array : array
        array of filenames, with etensions included.
    columns : int
        number of columns expected in the final datafile.

    Returns
    -------
    raw_data_array : numpy rray
        combiined data contained in each file.

    """
    data_array = np.zeros((1, columns))
    for file in filename_array:
        data_array = np.append(data_array, read_data(file), axis=0)
    raw_data_array = np.delete(combine_array(data_array), 0, axis=0)
    return raw_data_array


def remove_zero_values(input_array, column,
                       print_error_lines=PRINT_ERROR_LINES):
    """
    takes a 2D multi column float array and checks column {column} for zeros
    and outputs a float array of the lines with zeros omitted

    Parameters
    ----------
    input_array : 2D float array
        array of floats to be checked for zeros.
    column : int
        which column checked for zeros, index start at 0.
    print_error_lines : bool, optional
        prints which lines have zeros. The default is PRINT_ERROR_LINES.

    Returns
    -------
     : numpy array
        attenuated input array with lines with zeros in chosen column omitted.

    """
    checking_column = input_array[:, column]
    indexes_of_zeros = []
    for line in enumerate(checking_column):
        if line[1] == 0:
            if print_error_lines:
                print(f'line {line[0]} in the data array omitted due to '
                      f'zero in column {column}. {input_array[line[0],:]}')
            indexes_of_zeros.append(line[0])
    for index in sorted(indexes_of_zeros, reverse=True):
        input_array = np.delete(input_array, index, 0)
    return np.array(input_array)


def remove_negative_lines(input_array, column,
                          print_error_lines=PRINT_ERROR_LINES):
    """
    scans input arrays chosen column for negative values and removes the lines
    containting them from the input array

    Parameters
    ----------
    input_array : 2D array
        2d array to be scanned.
    column : int
        which column to be scanned, index from 0.
    print_error_lines : bool, optional
        prints flagged lines. The default is PRINT_ERROR_LINES.

    Returns
    -------
     : numpy array
        attenuated input array containg no negatives in chosen column.
    """
    checking_column = input_array[:, column]
    indexes_of_negatives = []
    for line in enumerate(checking_column):
        if line[1] < 0:
            if print_error_lines is True:
                print(f'line {line[0]} in the data array omitted due to '
                      f'negative value in column {column}.'
                      f' {input_array[line[0],:]}')
            indexes_of_negatives.append(line[0])
    for index in sorted(indexes_of_negatives, reverse=True):
        input_array = np.delete(input_array, index, 0)
    return np.array(input_array)


def make_column_positive(input_array, column):
    """
    takes a 2d array and makes the chosen column absolute

    Parameters
    ----------
    input_array : 2D array
        array with a column that needs to be positive.
    column : int
        column of array that needs to be converted to positive.

    Returns
    -------
    2D numpy array with chosen column that has become positive.

    """
    input_array = np.array(input_array)
    input_array[:, column] = np.abs(input_array[:, column])
    return input_array


def validate_data(input_array,
                  force_positive_uncertainty=FORCE_POSITIVE_UNCERTAINTY):
    """
    takes a 3 column input array with uncertainty in third column, makes
    uncertainty positive, checks for zero uncertainty and removes, and removes
    lines with non physica negative values

    Parameters
    ----------
    input_array : 2D array
        input array to check for validation.
    force_positive_uncertainty : bool, optional
        if true makes third column aboslute.
        The default is FORCE_POSITIVE_UNCERTAINTY.

    Returns
    -------
    positive_validation : TYPE
        DESCRIPTION.

    """
    if force_positive_uncertainty:
        input_array = make_column_positive(input_array, 2)
    zero_uncertainty_remove = remove_zero_values(input_array, column=2)
    positive_validation = np.array(zero_uncertainty_remove)
    for column_index in range(len(input_array[0])):
        positive_validation = remove_negative_lines(positive_validation,
                                                    column=column_index)
    return positive_validation


def find_average_of_datapoints(data_array, averaging_number, averaging_column):
    """
    takes a 2D array and for a chosen column will average for each datapoint
    the values of the surrounding datapoints

    Parameters
    ----------
    data_array : array
        data array where a column is to be averaged.
    averaging_number : int
        how many data points to average around each datapoint.
    averaging_column : int
        which column to be averaged.

    Returns
    -------
    average_array : numpy array
        array of similar dimension to data_array but averaged column.

    """
    average_array = np.array(data_array)
    averaging_number += 1
    averaging_array = data_array[:, averaging_column]
    number_datapoints = len(averaging_array)
    for index in enumerate(averaging_array):
        if index[0] in np.arange(0, averaging_number, 1):
            continue
        if index[0] in np.arange(number_datapoints-averaging_number,
                                 number_datapoints+1, 1):
            continue
        value_of_datapoints_around = []
        for number in range(1, averaging_number):
            value_of_datapoints_around.append(averaging_array[index[0]-number])
            value_of_datapoints_around.append(averaging_array[index[0]+number])
        average = np.mean(value_of_datapoints_around)
        average_array[index[0], averaging_column] = average
    return average_array


def remove_bulk_outliers(data_array, cutoff, averaging_number,
                         print_error_lines=PRINT_ERROR_LINES):
    """
    takes a data array, with odinate values in the 1st column, for each
    datapoint takes an average of the values around the datapoints and if
    the datapoint is cutoff times larger than the average it is counted as an
    outlier

    Parameters
    ----------
    data_array : 2d array
        with columns ordinate values in 2nd column.
    cutoff : int
        number of times greater a datapoint has to be compared to its average
        neighbours to be considered outlier
    averaging_number : int
        number of datapoints to average each side of the datapoint.
    print_error_lines : bool, optional
        if true ill print which lines in the data array are omitted.
        The default is PRINT_ERROR_LINES.

    Returns
    -------
    data_array : 2D array
        input array with outliers removed.

    """
    outlier_indexes = []
    average_array = find_average_of_datapoints(
        data_array, averaging_number, averaging_column=1)[:, 1]
    difference_array = data_array[:, 1]/average_array
    for index, difference in enumerate(difference_array):
        if difference > cutoff:
            if print_error_lines:
                print(f'datapoint {index} in data array more than '
                      f'{difference:g} times greater than surrounding values')
            outlier_indexes.append(index)
    for index in sorted(outlier_indexes, reverse=True):
        data_array = np.delete(data_array, index, 0)
    average_array = np.delete(average_array, 0, 0)
    if len(outlier_indexes) == 0:
        if PRINT_ERROR_LINES:
            print('no datapoints were removed during run')
    return data_array


def remove_outlier_using_uncertainty(data_array, cutoff, function, parameters,
                                     print_error_lines=PRINT_ERROR_LINES):
    """
    takes a 3 column data array with abcissa ordinate ordinate_uncertainty
    and a function to compare the datapoints to , the function evaulates the
    function at each absissa coordinate and calculates how far the pint is in
    multiples of standard deviations, if this value greater than cutoff then
    datapoint considered outlier

    Parameters
    ----------
    data_array : 2D 3 column array
        data array containting 3 columns.
    cutoff : int
        how many standard deviation away is point considered too far away.
    function : function two parameters
        function that takes two parameters.
    parameters : array like
        pairing of functions in order to be applied to the function.
    print_error_lines : bool, optional
        if true will print how many outliers removed. The default is
        PRINT_ERROR_LINES.

    Returns
    -------
    data_array : TYPE
        DESCRIPTION.

    """
    outlier_indexes = []
    for index, line in enumerate(data_array):
        deviation = ((line[1]
                      - function(line[0], parameters[0], parameters[1]))
                     / line[2])
        if np.abs(deviation) > cutoff:
            outlier_indexes.append(index)
    for index in sorted(outlier_indexes, reverse=True):
        data_array = np.delete(data_array, index, 0)
    if print_error_lines:
        print(f'{len(outlier_indexes)} points were removed for being over '
              f'{cutoff:.2f} standard deviations away from the graph that is'
              ' fit to the average of the datapoints')
    return data_array


def read_sort_validate_data(filename_array):
    """
    takes an array of filenames, opens each file for reading, reads the files
    and removes any non floats and nans and infs. then removes outliers by
    producing an average around each datapoint and comparing the value of the
    datapoint to the average, if datapoint too far different then removed.

    Parameters
    ----------
    filename_array : array of strings
        the names of the files to be read and combined, all files must contain
        the same number of columns.

    Returns
    -------
    natural_unit_data: numpy array
        the data array produced after reading the files and combining then
        sorting them, no nans.

    """
    raw_data_array = read_files_and_combine(filename_array, columns=3)
    if PRINT_ERROR_LINES:
        print('raw data array:', len(raw_data_array[:, 0]), 'datapoints\n')
    basic_validated_data = validate_data(raw_data_array)
    if PRINT_ERROR_LINES:
        print('after basic validation:', len(basic_validated_data[:, 0]),
              'datapoints\n')
    bulk_outlier_validated_data = remove_bulk_outliers(
        basic_validated_data, cutoff=3, averaging_number=2)
    if PRINT_ERROR_LINES:
        print('after bulk outlier removal:', len(
            bulk_outlier_validated_data[:, 0]), 'datapoints\n')
    fine_outlier_validated_data = remove_bulk_outliers(
        bulk_outlier_validated_data, cutoff=3, averaging_number=2)
    if PRINT_ERROR_LINES:
        print('after finer outlier removal:', len(
            fine_outlier_validated_data[:, 0]), 'datapoints\n')
    natural_unit_data = convert_array_units_nb_to_gev(
        fine_outlier_validated_data, columns=[1, 2])
    return natural_unit_data


def filter_data_by_average_fitting(
        natural_unit_data,
        minimisation_initial_guess):
    """
    minimises a fit to the average around each datapoint, and using that
    average fit function, the distance to each point is measured and if the
    distance is greater than the standard deviation cutoff its removed

    Parameters
    ----------
    natural_unit_data : array
        the data array that will have its second column averaged, two
        datapoints per side of the equation summed and divided.

    Returns
    -------
    final_data_natural_unit : array
        data array with outliers greater than 3 standard deviations removed.

    """
    average_array = find_average_of_datapoints(
        natural_unit_data, averaging_number=2, averaging_column=1)
    average_cross_sec_lambda = cross_section_chisquared_function(
        average_array, cross_section_function)
    average_result = minimise_two_parameter_function(
        average_cross_sec_lambda, minimisation_initial_guess)
    final_data_natural_unit = remove_outlier_using_uncertainty(
        natural_unit_data, 3, cross_section_function,
        [average_result[0], average_result[1]])
    if PRINT_ERROR_LINES:
        print('after final outlier removal:',
              len(final_data_natural_unit[:, 0]), 'datapoints\n')
    return final_data_natural_unit


# equation and units functions


def cross_section_function(com_energy, mass_z_boson, width_z_boson,
                           ee_partial_width=EE_PARTIAL_WIDTH):
    """
    all quantities in natural units
    models:
        cross_section = ((12 * pi)/mz^2
                         * (E^2 / (E^2 - mz^2)+mz^2 * widthZ^2)
                         * widthEE^2)
    function calculating the cross section for e+e- -> Z_0 -> e+e- interaction
    for a given com_energy, mass and width of z boson and ee_partial_width

    Parameters
    ----------
    com_energy : float
        centre of mass energy, independent variable GeV.
    mass_z_boson : float
        mass of the Z_0 boson GeV/c^2.
    width_Z_boson : float
        width of the Z_0 boson GeV.
    ee_partial_width : float, optional
        partial width of Z_0 -> e+e-. The default is EE_PARTIAL_WIDTH GeV.

    Returns
    -------
    cross_section : float
        cross section of interaction in GeV^-2.

    """
    temp_multiplier = (12*pc.pi)/(mass_z_boson**2)
    temp_denominator = (1/((com_energy**2 - mass_z_boson**2)**2
                        + (mass_z_boson**2 * width_z_boson**2)))
    cross_section = temp_multiplier*(com_energy**2 * temp_denominator)
    if EE_PARTIAL_WIDTH > 0:
        cross_section = cross_section * ee_partial_width**2
    return cross_section


def gev_to_milibarn(cross_section, invert=False):
    """
    convert from natural units where h=c=1 of cross section GeV
    (gigaelectronvolt) to milibarn

    Parameters
    ----------
    cross_section : float
        cross section in Gev.
    invert : bool optional
        if true reverses operation, mb to gev. The default is False.

    Returns


    Returns
    -------
    float
        corresponding cross section in mb.

    """

    multiplier = 197.33**2 / 10**5
    if invert:
        multiplier = 10**5 / 197.33**2
    return cross_section*multiplier


def nanobarn_to_gev(cross_section, invert=False):
    """
    converts cross section to nanobarn to gigaelectronvolt units if invert is
    False

    Parameters
    ----------
    cross_section : float
        cross section in nanobarn.
    invert : bool, optional
        if true revserses the operation. The default is False.

    Returns
    -------
    gev_crossection : float
        when invert is false, returns the cross section in its natural units.
    nanobarn_crossection : float
        when invert is true, returns cross section in nanobarn units

    """
    if not invert:
        millibarn_cross_section = cross_section * 10**-6
        gev_crossection = gev_to_milibarn(millibarn_cross_section,
                                          invert=True)
        return gev_crossection
    milibarn_crossection = gev_to_milibarn(cross_section, invert=False)
    nanobarn_crossection = milibarn_crossection * 10**6
    return nanobarn_crossection


def convert_array_units_nb_to_gev(data_array, columns, invert=False):
    """
    takes a list of columns and a data array, converts the chosen columns
    from nanobarn units to gev units

    Parameters
    ----------
    data_array : 2D array
        data array with columns in gev.
    columns : array like
        list of integers, chosen columns to convert.
    invert : bool, opional
        if true array chosen column go from gev to nb units

    Returns
    -------
    data_array : numpy array
        the data array after converting chosen column units.

    """
    data_array = np.array(data_array)
    if invert is True:
        for column in columns:
            data_array[:, column] = nanobarn_to_gev(data_array[:, column],
                                                    invert=True)
    else:
        for column in columns:
            data_array[:, column] = nanobarn_to_gev(data_array[:, column])
    return data_array


def lifetime_z_boson_calculate(width_z_boson):
    """
    calculates the lifetime of Z boson given its width in GeV, first
    converts to SI units
    models:
        hbar / widthZ


    Parameters
    ----------
    width_Z_boson : float
        width of Z boson, GeV

    Returns
    -------
    float
        lifetime in seconds

    """
    width_z_boson_si_units = width_z_boson * pc.e * 10**9
    return pc.hbar/width_z_boson_si_units


def calculate_reaction_rate(cross_section, beam_luminosity=BEAM_LUMINOSITY):
    """
    calculates the reaction rate based on R = cross_section * beam_luminosity
    by first converting cross section to barn^2 units

    Parameters
    ----------
    cross_section : flaot
        value of cross section in [GeV^-2].
    beam_luminosity : float, optional
        value of beam luminosity in [cm^-2 s^-1].
        The default is BEAM_LUMINOSITY.

    Returns
    -------
    reaction_rate : float
        rate of event in [s^-1].

    """
    beam_luminosity = beam_luminosity * 100**2  # m^-2s^-1
    milibarn_cross_section = gev_to_milibarn(cross_section, invert=False)
    barn_cross_section = milibarn_cross_section/1000
    metre_cross_section = barn_cross_section * 10**-28
    reaction_rate = metre_cross_section * beam_luminosity
    return reaction_rate
# chi_squared_minimisation_functions


def calculate_chi_squared(observation, observation_uncertainty, prediction):
    """
    function that calculates the chi squared of a function with respect to some
    observed datapoints and their corresponding uncertainties

    Parameters
    ----------
    observation : 1D array
        array.
    observation_uncertainty : 1D array
        array of uncertainties coupled to observation.
    prediction : function
        theoretical calculator for what obersvation should be at some abscissa.


    Returns
    -------
    chi_squared : float
        value of chi squared.
    """
    chi_squared = np.sum((observation-prediction)**2
                         / (observation_uncertainty)**2)
    return chi_squared


def cross_section_chisquared_function(input_data, function):
    """
    function takes in a data array and funtion and returns a function of
    purley two parameters - that function will calculate the chi squared for
    a certain value of chi squared

    Parameters
    ----------
    input_data : 2D float array
        0 column is the independednt variable
        1 column is the observed values
        2 column is the uncertainty

    Returns
    -------
    lambda function
        function that takes in two parameters and outputs chi squared and
        number of datapoints.

    """
    return (lambda parameter_one, parameter_two:
            calculate_chi_squared(observation=input_data[:, 1],
                                  observation_uncertainty=input_data[:, 2],
                                  prediction=function(input_data[:, 0],
                                                      parameter_one,
                                                      parameter_two)))


def minimise_two_parameter_function(function, initial_guess):
    """
    simple algorithm to minimise a function

    Parameters
    ----------
    function : function of two parameters
        two parameter function, to be minimised.
    initial_guess : arraylike
        1d array with 2 elements, initial guess for the minimisation.

    Returns
    -------
    a: float
        minimised value of first parameter.
    b: float
        minimised value of second parameter.
    min_function : float
        minimum value of the function.
    counter : int
        number of iterations took to obtain the value.

    """
    step_size = MINIMISATION_STEP_SIZE
    tolerance = MINIMISATION_TOLERANCE
    max_counter = MINIMISATION_MAX_COUNTER

    a = initial_guess[0]  # first parameter
    b = initial_guess[1]  # second parameter
    counter = 0
    difference = 1
    min_function = function(a, b)  # initial

    while difference > tolerance:
        counter += 1
        difference_array = np.array([])
        function_evaluation = [function(a+step_size, b+step_size),
                               function(a+step_size, b-step_size),
                               function(a+step_size, b),
                               function(a-step_size, b+step_size),
                               function(a-step_size, b-step_size),
                               function(a-step_size, b),
                               function(a, b+step_size),
                               function(a, b-step_size)]
        for element in function_evaluation:
            difference_temp = element-min_function
            difference_array = np.append(difference_array, difference_temp)
        most_negative_index = np.argmin(difference_array)
        if difference_array[most_negative_index] > 0:
            step_size *= 0.5
        elif most_negative_index == 0:
            a += step_size
            b += step_size
            min_function = function(a, b)
        elif most_negative_index == 1:
            a += step_size
            b -= step_size
            min_function = function(a, b)
        elif most_negative_index == 2:
            a += step_size
            min_function = function(a, b)
        elif most_negative_index == 3:
            a -= step_size
            b += step_size
            min_function = function(a, b)
        elif most_negative_index == 4:
            a -= step_size
            b -= step_size
            min_function = function(a, b)
        elif most_negative_index == 5:
            a -= step_size
            min_function = function(a, b)
        elif most_negative_index == 6:
            b += step_size
            min_function = function(a, b)
        elif most_negative_index == 7:
            b -= step_size
            min_function = function(a, b)
        if counter == max_counter:
            print('run out of counter minimising the function')
            break
        difference = np.abs(difference_array[most_negative_index])
    return a, b, min_function, counter

# finding uncertainty functions


def vary_parameter_until_target(function, minimised_parameters, plus_value,
                                flip_step=False, flip_parameter=False):
    """
    function that varies a parameter until the function it is fed into returns
    a value within tolerance of a target value decided by a value in minimised
    parameters + plus value. flip direction of step or which parameter is
    varied

    Parameters
    ----------
    function : function
        function that takes two parameters, data array already inbuilt.
    minimised_parameters : array like
        values of minimised parameters and function value at the minimsed
        parameter.
    plus_value : float
        what value added to make target value.
    flip_step : bool, optional
        if true varies the parameter in the opposite direction.
        The default is False.
    flip_parameter : bool, optional
        if true varies the second parameter. The default is False.

    Returns
    -------
    parameter_one : float
        when flip parameter false, returns the value of parameter that returns
        a value of the function close to target value.
        whenf flip parameter true, returns the same value as inputted
    parameter_two : float
        when flip parameter true returns the value of the parameter that makes
        the function go to target value within tolerance.
    new_function_value : float
        the final value of the function such that it is within tolerance of
        target(if programm ran correctly)
    difference : float
        a value that should be below set tolerance if program ran correctly.

    """
    step_size = 0.000005
    if flip_step:
        step_size *= -1
    tolerance = 0.001
    max_counter = 100000
    target_value = minimised_parameters[2]+plus_value
    parameter_one = minimised_parameters[0]
    parameter_two = minimised_parameters[1]
    difference = 1
    counter = 0
    while difference > tolerance:
        counter += 1
        if not flip_parameter:
            parameter_one += step_size
        elif flip_parameter:
            parameter_two += step_size
        new_function_value = function(parameter_one, parameter_two)
        difference = new_function_value-target_value
        if difference > 0:
            step_size *= -0.7
        difference = np.abs(difference)
        if counter == max_counter:
            print('counter ran out varying parameter to target value')
            break
    return parameter_one, parameter_two, new_function_value, difference


def calculate_sigma(function, minimised_parameters, plus_value):
    """
    function that applied the vary_parameter_until_value function and finds the
    higher and lower value each parameter can be to make the function go to
    target value, and divides it by 2 to find the distance in parameterspace to
    get from a minimised value to the target value

    Parameters
    ----------
    function : function
        two parameter function with data array already inbuilt.
    minimised_parameters : array like
        array containting the minimised value of parameters and the value of
        the function when evaulated at the minimised parameterspace location.
    plus_value : float
        what value is added to the minimised function value, target value.

    Returns
    -------
    sigma_one : float
        average distance in parameterspace the fist parameter has to travel to
        force function to the target value.
    sigma_two : float
        average distance in parameterspace the second parameter has to travel
        to force function to the target value.

    """
    result_one_plus = vary_parameter_until_target(
        function, minimised_parameters, plus_value,
        flip_step=False, flip_parameter=False)
    result_one_minus = vary_parameter_until_target(
        function, minimised_parameters, plus_value,
        flip_step=True, flip_parameter=False)

    result_two_plus = vary_parameter_until_target(
        function, minimised_parameters, plus_value,
        flip_step=False, flip_parameter=True)
    result_two_minus = vary_parameter_until_target(
        function, minimised_parameters, plus_value,
        flip_step=True, flip_parameter=True)

    sigma_one = (result_one_plus[0]-result_one_minus[0])*0.5
    sigma_two = (result_two_plus[1]-result_two_minus[1])*0.5
    return sigma_one, sigma_two


def calculate_lifetime_sigma(lifetime, width, sigma_width):
    """
    simply calculates the uncertainty on the lifetime by finding the fractional
    uncertainty and propogating that to the calculated lifetime

    Parameters
    ----------
    lifetime : float
        the value of z boson lifetime.
    width : float
        the value of z boson width.
    sigma_width : float
        the uncertainty on z boson width.

    Returns
    -------
    sigma_lifetime : float
        the uncertainty on the z boson lifetime.

    """
    sigma_lifetime = lifetime * sigma_width/width
    return sigma_lifetime


def find_array_limits(data_array, column=0):
    """
    takes an array input and checks if its 1 or 2D, then find the minimum and
    maximum of the array, column can be chosen if a 2D array is input

    Parameters
    ----------
    data_array : array
        the input array of floats that the min and max values need to be found.
    column : int, optional
        whihc column in a 2D array to find the min and max of.
        The default is 0.

    Returns
    -------
    minimum_value : float
        the minimum value in the array chosen column.
    maximum_value : float
        the maximum value in the array chosen column.

    """
    try:
        number_columns = len(data_array[0])
    except TypeError:
        number_columns = 1
    if number_columns == 1:
        value_array = data_array
    else:
        value_array = data_array[:, column]
    minimum_value = np.amin(value_array)
    maximum_value = np.amax(value_array)
    return minimum_value, maximum_value


def mesh_couple_two_arrays(x_array, y_array):
    """
    takes two 1d arrays as inputs and returns a regular x mesh, y mesh and
    then a coupled mesh that contains a tuple at each index containting the
    pairing of the x and y individual meshes

    Parameters
    ----------
    x_array : 1D array
        array of values to mesh, x coordinate values.
    y_array : 1D array
        array of values to mesh, y coordinate values.

    Returns
    -------
    pair_array : 2D array of tuples
        array of tuples containting the (x,y) coordinate at each point.
    x_mesh : 2D array of floats
        an array containing 1 uniqe row duplicated len(y_array) times.
    y_mesh : 2D array of floats
        an array containing 1 uniqe column duplicated len(x_array) times.

    """
    x_mesh, y_mesh = np.meshgrid(x_array, y_array)
    pair_array = np.zeros((len(y_array), len(x_array)), dtype=object)
    for row, y_value in enumerate(sorted(y_array, reverse=True)):
        for column, x_value in enumerate(x_array):
            pair_array[row, column] = np.array([x_value, y_value])
    return pair_array, x_mesh, y_mesh


def evaluate_function_over_array(function, parameter_array):
    """
    takes a two parameter function and an array of parameterspace tuple
    coordiantes and produces an array of similar shape to paramter_array except
    the value at each point is the function evaulated at the coordinate pair

    Parameters
    ----------
    function : function
        two parameter function that will be evaluated over parameterspace.
    parameter_array : 2D array of tuples
        a 2D array containting the coordinate values of each point to be
        evaluated.

    Returns
    -------
    evaluated_array : TYPE
        DESCRIPTION.

    """
    evaluated_array = np.zeros((np.shape(parameter_array)))
    shape = np.shape(evaluated_array)
    for row in range(shape[0]):
        for column in range(shape[1]):
            evaluated_array[row, column] = function(
                parameter_array[row, column][0],
                parameter_array[row, column][1])
    return evaluated_array


def plot_datapoints_and_function(data_array, two_parameter_function,
                                 parameters, function_resolution=0.0001):
    """
    plots the datapoints and a corresponding function that takes two parameters
    as input values, plots the curve in the left two thirds of the figure, and
    also annotates the values for each parameter and fit goodness values. also
    plots the full width half max if user sets PLOT_FWHM to true

    Parameters
    ----------
    data_array : array like
        an array containting the ordinate and abscissa values and their
        corresponding uncertainties.
    two_parameter_function : function
        function that has two parameters as input, a data array is generated
        over the data arrays abscissa range (0th column of data array).
    parameters : array like
        2 element 1d array, contains the parameters to input to the
        function to plot.
    function_resolution : float, optional
        step size of abscissa. The default is 0.0001.

    Returns
    -------
    None.

    """
    min_abscissa, max_abscissa = find_array_limits(data_array[:, 0])
    abscissa_array = np.arange(min_abscissa, max_abscissa, function_resolution)
    ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax.errorbar(data_array[:, 0], data_array[:, 1], yerr=data_array[:, 2],
                fmt='x', markersize=3, elinewidth=0.5, capsize=1, capthick=0.5,
                color=GRAPH_COLOR_DATAPOINT,
                label=f'datapoints ({len(data_array[:,0])})')

    ax.plot(abscissa_array,
            two_parameter_function(abscissa_array,
                                   parameters[0], parameters[1]),
            linewidth=0.7, color=GRAPH_COLOR_FUNCTION,
            label='predicted Z-boson function\n'+(r'$\sigma = '
                                                  r'\dfrac{12\pi}{m_z^2}'
                                                  r'\dfrac{E^2}{(E^2-m_z^2)^2'
                                                  r'+m_z^2\Gamma_z^2}'
                                                  r'\Gamma_{ee}^2$'))
    # find fwhm
    function_array = np.column_stack((abscissa_array, two_parameter_function(
        abscissa_array, parameters[0], parameters[1])))
    full_width_half_max = find_full_width_fraction_maxima(function_array)

    if PLOT_FWHM:
        start_energy = full_width_half_max[1][0]
        end_energy = full_width_half_max[1][1]
        value = full_width_half_max[1][2]
        ax.axhline(y=value, color=FWHM_COLOR, linewidth=0.7,
                   linestyle=FWHM_LINESTYLE, alpha=FWHM_ALPHA,
                   label='full width half maximum 'r'($\sigma_{FWHM}$)')
        ax.vlines(x=start_energy, ymin=0, ymax=value, color=FWHM_COLOR,
                  linewidth=0.7, linestyle=FWHM_LINESTYLE,
                  alpha=FWHM_ALPHA)
        ax.vlines(x=end_energy, ymin=0, ymax=value, color=FWHM_COLOR,
                  linewidth=0.7, linestyle=FWHM_LINESTYLE,
                  alpha=FWHM_ALPHA)
    if PLOT_MAXIMISING_ENERGY:
        ax.axvline(x=parameters[11][1], color=ENERGY_LINE_COLOR,
                   alpha=ENERGY_LINE_ALPHA, linewidth=0.7,
                   linestyle=ENERGY_LINE_LINESTYLE,
                   label=r'maximising energy ($E_{maximising}$)')

    ax.grid(True)
    ax.set_xlabel(r'Energy GeV', fontsize=11)
    ax.set_ylabel(r'cross section $\sigma$ GeV$^{-2}$', fontsize=11)
    ax.set_title('cross section as a function of energy in natural units',
                 fontsize=12)

    # display equation
    ax.annotate((r'$R$ = $\sigma$ $\cdot$ $L$'),
                xy=(1.05, 0.5), xytext=(0, 0), xycoords='axes fraction',
                textcoords='offset points', fontsize=10)
    ax.annotate((r'$\tau_Z$ = $\hbar/\Gamma_Z$'),
                xy=(1.05, 0.5), xytext=(60, 0), xycoords='axes fraction',
                textcoords='offset points', fontsize=10)
    ax.annotate(('Given Values:'),
                xy=(1.03, 0.49), xytext=(0, -11), xycoords='axes fraction',
                textcoords='offset points', fontsize=10, style='oblique')
    ax.annotate((r' $\Gamma_{ee} = 'rf'{parameters[5]:.4g}$ GeV'),
                xy=(1.04, 0.49), xytext=(0, -22), xycoords='axes fraction',
                textcoords='offset points', fontsize=10)
    ax.annotate((r' $L$ = 'rf'{BEAM_LUMINOSITY:.2e} cm$^-$$^2$ s$^-$$^1$ '),
                xy=(1.04, 0.49), xytext=(0, -33), xycoords='axes fraction',
                textcoords='offset points', fontsize=10)

    # display parameter values
    ax.annotate(('Calculated Values:'),
                xy=(1.03, 0.33), xytext=(0, 12), xycoords='axes fraction',
                textcoords='offset points', fontsize=10, style='oblique')
    ax.annotate((fr'$m_Z = {parameters[0]:.3e}$'
                 + r' $\pm$' + f' {parameters[6]:1.0e}' + r' GeV/c$^2$'),
                xy=(1.05, 0.33), xytext=(0, 0), xycoords='axes fraction',
                textcoords='offset points', fontsize=10)
    ax.annotate((r' $\Gamma_Z = 'rf'{parameters[1]:.3e}$'
                 + r' $\pm$' + f' {parameters[7]:1.1e}' + ' GeV'),
                xy=(1.05, 0.33), xytext=(0, -11), xycoords='axes fraction',
                textcoords='offset points', fontsize=10)
    ax.annotate((r' $\tau_Z = 'rf'{parameters[4]:.2e}$' + r' $\pm$'
                 + f' {parameters[8]:1.0e}' + ' s'),
                xy=(1.05, 0.33), xytext=(0, -23), xycoords='axes fraction',
                textcoords='offset points', fontsize=10)
    ax.annotate((r'$E_{maximising}$ = 'f'{parameters[11][1]:.3e} GeV'),
                xy=(1.05, 0.33), xytext=(0, -35), xycoords='axes fraction',
                textcoords='offset points', fontsize=10)
    ax.annotate((r'$R_{max}$ = 'f'{parameters[11][0]:.2e} ' + r' s$^-$$^1$'),
                xy=(1.05, 0.33), xytext=(0, -47), xycoords='axes fraction',
                textcoords='offset points', fontsize=10)
    ax.annotate((r'$\sigma_{max}$ = 'f'{parameters[11][2]:.2e} '
                 + r' GeV$^-$$^2$'),
                xy=(1.05, 0.33), xytext=(0, -59), xycoords='axes fraction',
                textcoords='offset points', fontsize=10)
    ax.annotate((r'$\sigma_{FWHM}$ = 'f'{full_width_half_max[1][2]:.2e} ' +
                 'GeV/c$^-$$^2$'),
                xy=(1.05, 0.33), xytext=(0, -71), xycoords='axes fraction',
                textcoords='offset points', fontsize=10)
    ax.annotate((r'$E_{FWHM}$ = 'f'{full_width_half_max[0]:.3f} ' + 'GeV'),
                xy=(1.05, 0.33), xytext=(0, -83), xycoords='axes fraction',
                textcoords='offset points', fontsize=10)
    # chi squared values
    ax.annotate(('Fit Goodness Values:'),
                xy=(1.03, 0.035), xytext=(0, 12), xycoords='axes fraction',
                textcoords='offset points', fontsize=10, style='oblique')
    ax.annotate((r'$\chi^2_{min}$ ='fr' {parameters[2]:.4g} '),
                xy=(1.05, 0.035), xytext=(0, 0), xycoords='axes fraction',
                textcoords='offset points', fontsize=10)
    ax.annotate((f'd.o.f = {len(data_array[:,0])-2}'),
                xy=(1.05, 0.035), xytext=(-4, -12), xycoords='axes fraction',
                textcoords='offset points', fontsize=10)
    ax.annotate((r'$\chi^2_R$ ='fr' {parameters[3]:.3f} '),
                xy=(1.05, 0.035), xytext=(0, -24), xycoords='axes fraction',
                textcoords='offset points', fontsize=10)
    ax.legend(loc='upper left')


def produce_chi_contour_plot(lambda_chi_squared_function, minimised_parameters,
                             fig):
    """
    takes the minimised parameter and chi squared, plots the contour plot to
    show the parameter space chi squared plot. plots the graph in the top
    right of the figure

    Parameters
    ----------
    lambda_chi_squared_function : function
        function that calculates the chi squared for a pair of parameters.
    minimised_parameters : array like
        1d array where the first two elements are the parameters that minimise
        chi squared.
    fig :  matplotlib.figure.Figure
        the figure element the plot is to be added to.

    Returns
    -------
    None.

    """
    x_middle = minimised_parameters[0]  # mass
    y_middle = minimised_parameters[1]  # width
    x_values = np.arange(x_middle*0.9995, x_middle*1.0005, 0.001)
    y_values = np.arange(y_middle*0.985, y_middle*1.015, 0.001)
    pair_mesh, x_mesh, y_mesh = mesh_couple_two_arrays(x_values, y_values)
    function_values = evaluate_function_over_array(lambda_chi_squared_function,
                                                   pair_mesh)
    ax = plt.subplot2grid((8, 8), (0, 6), colspan=2, rowspan=3)
    contour_levels = [minimised_parameters[2]+1,
                      minimised_parameters[2]+2.3,
                      minimised_parameters[2]+5.99]
    contour_plot = ax.contour(x_mesh, y_mesh, function_values,
                              levels=contour_levels,
                              colors=CONTOUR_COLOR)
    ax.clabel(contour_plot, fmt='%.4g', inline=1)
    filled_contour = ax.contourf(x_mesh, y_mesh, function_values, 100,
                                 cmap=CONTOUR_STYLE)
    fig.colorbar(filled_contour)
    ax.set_xlabel('Mass $m_Z$ GeV/c$^2$', fontsize=10, )
    plt.xticks(rotation=0, fontsize=8)
    ax.set_ylabel(r'Width $\Gamma_Z$ GeV', fontsize=10)
    ax.set_title(r'chi squared $\chi^2$ contour plot,'
                 r' 1$\sigma$, 2$\sigma$, 3$\sigma$',
                 fontsize=11)


def print_runtime_message(start_time, calculation_time):
    """
    prints a message telling the user how long each section took to run

    Parameters
    ----------
    start_time : float
        start time of program.
    calculation_time : float
        time taken to calculate minimsed fit and uncertainties.

    Returns
    -------
    None.

    """
    print('\nFiler/Calculation runtime: '
          f'{calculation_time -start_time :.3f} seconds')
    print('Graphing runtime: '
          f'{timer.time() - calculation_time:.3f} seconds')
    print('Total runtime: '
          f'{(timer.time()-start_time):.3f} seconds')


def print_results(parameters):
    """
    prints values

    Parameters
    ----------
    parameters : array
        length of at least 10 array of floats and integers to print.

    Returns
    -------
    None.

    """
    print(f'minimising mass zboson: {parameters[0]:.3e} ± '
          f'{parameters[6]:1.0e} GeV/c^2')
    print(f'minimising width zboson: {parameters[1]:.3e} '
          f'± {parameters[7]:1.1e} GeV')
    print(f'lifetime zboson: {parameters[4]:.2e} ± {parameters[8]:1.0e} s')
    print(f'min chi squared: {parameters[2]:.4g}')
    print(f'reduced chi squared: {parameters[3]:.3f}')
    print(f'minimisation iterations: {parameters[9]}')
    print(f'number of datapoints: {parameters[10]}')
    print(f'maximising energy: {parameters[11][1]:.3e} GeV')
    print(f'maximum cross section: {parameters[11][2]:.3e} GeV^-2')
    print(f'instantanous beam luminsoity: {BEAM_LUMINOSITY:.2e} cm^-2 s^-1')
    print(f'maximum event rate: {parameters[11][0]:.2e} s^-1 ')
    print('The FWHM values are available on the figure. ')


def find_peak_function(data_array, function, parameters,
                       function_resolution=0.0001):
    """
    takes a data array and a corresponding function modeling the data, and
    finds what absicssa coordinates the data array is over, and produces finds
    the maxima of the function in that range

    Parameters
    ----------
    data_array : array
        2D array containting the abscissa in the first column.
    function : function of two parameters
        two parameter function to model data.
    parameters : array like
        values of the good fit parameters.
    function_resolution : float, optional
        step size between absicca values in function. The default is 0.0001.

    Returns
    -------
    max_ordinate : float
        maximum value of the function over the range of data.
    maximising_abscissa : float
        what value of abscissa maximises the function.

    """
    min_abscissa, max_abscissa = find_array_limits(data_array[:, 0])
    abscissa_array = np.arange(min_abscissa, max_abscissa, function_resolution)
    ordinate_array = function(abscissa_array, parameters[0], parameters[1])
    max_ordinate = np.max(ordinate_array)
    maximising_abscissa = abscissa_array[np.argmax(ordinate_array)]
    return max_ordinate, maximising_abscissa


def calculate_peak_reaction_rate(data_array, function, parameters):
    """
    calculates the maximum rate of event occurence based on the value of
    maximum cross section, for some beam luminosity provided

    Parameters
    ----------
    data_array : array
        2D array containting the abscissa in the first column.
    function : function of two parameters
        two parameter function to model data.
    parameters : array like
        values of the good fit parameters.

    Returns
    -------
    max_reaction_rate : float
        the maximum reaction rate based on the maximum cross section and
        beam luminosity in s^-1.
    maximising_energy : float
        the enrgy where maximum cross section occurs in gev.
    max_cross_section : float
        the value of maximum cross section in gev^-2.

    """
    max_cross_section, maximising_energy = find_peak_function(data_array,
                                                              function,
                                                              parameters)
    max_reaction_rate = calculate_reaction_rate(max_cross_section)
    return max_reaction_rate, maximising_energy, max_cross_section


def find_full_width_fraction_maxima(data_array, fraction=0.5):
    """
    takes a data array and find the full width (half) fraction maxima, i.e the
    distance between the abscissa values where the ordinate values are closest
    to the fraction * max value

    Parameters
    ----------
    data_array : array
        at least two columns.
    fraction : float, optional
        what value of full width maxima to find. The default is 0.5.

    Returns
    -------
    width : float
        the value of the full width fraction maxima.

    """
    if fraction > 1:
        print('the fraction in full width fraction maxima must be less than 1')
    max_value = np.max(data_array[:, 1])
    index_of_max_value = np.argmax(data_array[:, 1])
    comparator_value = max_value*fraction
    difference_array = np.abs(data_array[:, 1]-comparator_value)
    energy_difference_array = np.column_stack((data_array[:, 0],
                                               difference_array))
    pre_maxima_array = energy_difference_array[0:index_of_max_value, :]
    post_maxima_array = energy_difference_array[index_of_max_value:, :]
    sorted_pre_max = np.array(sorted(pre_maxima_array, key=lambda x: x[1]))
    sorted_post_max = np.array(sorted(post_maxima_array, key=lambda x: x[1]))
    width = sorted_post_max[0][0]-sorted_pre_max[0][0]
    return width, (sorted_pre_max[0][0], sorted_post_max[0][0],
                   comparator_value)


def find_initial_guess(validated_data_array):
    """
    finds an inital guess bases on the peak value and full width half maxima
    of the averaged input data array

    Parameters
    ----------
    validated_data_array : array
        data array to find the full width half maxima.

    Returns
    -------
    mass_guess : float
        an initial guess on the mass corresponing to maximising energy.
    width_guess : float
        an initial guess on width corrsponding to full width maxima of points.

    """
    average_array = find_average_of_datapoints(validated_data_array,
                                               averaging_number=2,
                                               averaging_column=1)
    mass_guess = average_array[:, 0][np.argmax(average_array[:, 1])]
    width_guess = find_full_width_fraction_maxima(average_array)
    return mass_guess, width_guess[0]


def initial_guess_choose(natural_unit_data):
    """
    takes a data array, and if AUTO_INITIAL_GUESS is true then finds an inital
    guess, otherwise uses provided inital guess

    Parameters
    ----------
    natural_unit_data : array
        validated data array .

    Returns
    -------
    initial_guess : array
        two element list, with [mass initial guess, width initial guess] in
        natural units.

    """
    if AUTO_INITAL_GUESS:
        print('automatically finding an initial guess..')
        initial_guess = find_initial_guess(natural_unit_data)
        print(f'inital guess found!: \n -mass ~ {initial_guess[0]:.2f} GeV/c^2'
              f'\n -width ~ {initial_guess[1]:.2f} GeV\n')
    else:
        initial_guess = MINIMISATION_INITIAL_GUESS
        print('using initial guess provided:')
        print(f'initial guess found!: \n -mass ~ {initial_guess[0]:.2f} '
              f'GeV/c^2\n -width ~ {initial_guess[1]:.2f} GeV\n')
    return initial_guess


def print_welcome_message():
    """
    prints a message to the console for the user

    Returns
    -------
    None.

    """
    print("Z-BOSON".center(50, "-"))
    print('Hello, this code plots the datapoints from a particle physics'
          ' experiment and finds the corresponding parameters that minimise'
          ' the LSQ')
    print('By looking at the constants at the start of the code you can:'
          '\n - Add or change the analysed datafiles, note: the code runs for'
          ' an arbitrary number of files'
          '\n - Figure saving and graph colour/ style options'
          '\n - Automatically find an initial guess'
          '\n - Change the BEAM_LUMINOSITY used to calculate the maximum event'
          ' rate')


def main():
    """
    main function that calls the functions in the correct order, filer data,
    calculate minimsed fit, find necessary values, plot graph and measures the
    time to complete each section.

    Returns
    -------
    None.

    """
    start_time = timer.time()
    print_welcome_message()
    # read, sort and validate data
    print("\n"+"Filtering The Data".center(50, "-"))
    print(f'reading and filtering data... : {FILES}')
    natural_unit_data = read_sort_validate_data(FILES)
    # inital guess
    initial_guess = initial_guess_choose(natural_unit_data)
    # final filtering
    final_data_natural_unit = filter_data_by_average_fitting(
        natural_unit_data,
        minimisation_initial_guess=initial_guess)
    # obtain parameter values (minimisation)
    cross_sec_chisquared_lambda = cross_section_chisquared_function(
        final_data_natural_unit, cross_section_function)
    mass_width_min_chisquared_counter = minimise_two_parameter_function(
        cross_sec_chisquared_lambda, initial_guess)
    reduced_chi_squared = (mass_width_min_chisquared_counter[2]
                           / (len(final_data_natural_unit[:, 0])-2))
    print('data filtered!')
    print('\ncalculating best fit...')

    # calculate lifetime
    lifetime = lifetime_z_boson_calculate(mass_width_min_chisquared_counter[1])

    info_parameters = [mass_width_min_chisquared_counter[0],
                       mass_width_min_chisquared_counter[1],
                       mass_width_min_chisquared_counter[2],
                       reduced_chi_squared, lifetime, EE_PARTIAL_WIDTH]
    peak_rate = calculate_peak_reaction_rate(final_data_natural_unit,
                                             cross_section_function,
                                             info_parameters)
    # calculate uncertainties
    mass_width_sigma = calculate_sigma(
        cross_sec_chisquared_lambda, info_parameters, plus_value=1)
    sigma_lifetime = calculate_lifetime_sigma(
        lifetime, mass_width_min_chisquared_counter[1], mass_width_sigma[1])

    info_parameters = [mass_width_min_chisquared_counter[0],
                       mass_width_min_chisquared_counter[1],
                       mass_width_min_chisquared_counter[2],
                       reduced_chi_squared, lifetime, EE_PARTIAL_WIDTH,
                       mass_width_sigma[0], mass_width_sigma[1],
                       sigma_lifetime, mass_width_min_chisquared_counter[3],
                       len(final_data_natural_unit[:, 0]), peak_rate]
    print('values found!')
    # results
    print("\n"+"Results and Values".center(50, "-"))
    print_results(info_parameters)
    calculation_time = timer.time()
    # plot data
    print('\nplotting...')
    plt.style.use(GRAPH_STYLE)
    fig = plt.figure(num=1, figsize=(12.4, 7), layout='tight')
    plt.tight_layout()
    plot_datapoints_and_function(
        final_data_natural_unit, cross_section_function, info_parameters)
    produce_chi_contour_plot(cross_sec_chisquared_lambda,
                             info_parameters, fig)
    if SAVE_FIGURE:
        plt.savefig(FIGURE_NAME, dpi=600, bbox_inches='tight')
        print(f'figure saved! as: {FIGURE_NAME} to current directory')
    plt.show()
    plt.close(fig=1)
    # print runtime
    print_runtime_message(start_time, calculation_time)
    return 0
# return 1 2 3 .. etc for sections in main that fail, exit niceley and reason
# known


if __name__ == '__main__':
    main()
