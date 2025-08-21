import numpy as np
from scipy.optimize import curve_fit

def detrend(time_series, amplitudes, uncertainties=None):
    '''This removes any constant or linear trend in the data series and makes sure the
    amplitudes are stored as a numpy array.

    :param time_series: list or np.array of values representing the observation times
    :param amplitudes: list or np.array of values repreenting the amplitude at
        each point in time
    :param uncertainties: None, list, or np.array of values representing the
        uncertainty in amplitude at each position

    :return: np.array of detrended amplitudes at each original time
    '''

    if not isinstance(amplitudes, np.ndarray):
        amplitudes = np.asarray(amplitudes)
    if not isinstance(time_series, np.ndarray):
        time_series = np.asarray(time_series)
    
    linear_fit = lambda x, a, b: a*x + b

    params, _ = curve_fit(
        linear_fit,
        time_series,
        amplitudes,
        p0=(
            (amplitudes[-1] - amplitudes[0]) / (time_series[-1] - time_series[0]),
            np.mean(amplitudes)
        ),
        sigma=uncertainties,
    )

    amplitudes -= linear_fit(time_series, params[0], params[1])

    return amplitudes


def set_reference_time(time_series_1, time_series_2):
    '''This defines the reference time as the smallest time in either time
    series, then shifts all times to start from this reference time and assures
    the output time series are stored as numpy arrays.

    :param time_series_1: list or np.array of values representing the observation times
    :param time_series_2: list or np.array of values representing the observation times

    :return: two instances of np.array which shift the starting point to
        the minimum time
    '''

    if not isinstance(time_series_1, np.ndarray):
        time_series_1 = np.asarray(time_series_1)
    if not isinstance(time_series_2, np.ndarray):
        time_series_2 = np.asarray(time_series_2)

    min_time = np.min(
        np.concatenate((time_series_1, time_series_2))
    )
    
    time_series_1 -= min_time
    time_series_2 -= min_time

    return time_series_1, time_series_2


def prep_time_series(
    time_series_1,
    amplitudes_1,
    time_series_2,
    amplitudes_2,
    uncertainties_1=None,
    uncertainties_2=None
):
    '''This prepares the time series data for computing the discrete
    correlation function.

    :param time_series_1: list or np.array of values representing the observation
        times in the first signal
    :param amplitudes_1: list or np.array of values repreenting the amplitude at
        each point in time for the first signal
    :param time_series_2: list or np.array of values representing the observation
        times in the first signal
    :param amplitudes_2: list or np.array of values repreenting the amplitude at
        each point in time for the second signal
    :param uncertainties_1: None, list, or np.array of values representing the
        uncertainty in amplitude at each position in the first time series
    :param uncertainties_2: None, list, or np.array of values representing the
        uncertainty in amplitude at each position in the second time series

    :return: multiple instances of np.array objects representing the prepared time
        series and amplitudes for each signal
    '''
    
    time_series_1, time_series_2 = set_reference_time(
        time_series_1,
        time_series_2
    )
    
    amplitudes_1 = detrend(time_series_1, amplitudes_1, uncertainties=uncertainties_1)
    amplitudes_2 = detrend(time_series_2, amplitudes_2, uncertainties=uncertainties_2)

    return time_series_1, amplitudes_1, time_series_2, amplitudes_2
















