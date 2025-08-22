import numpy as np
from modern_dcf.utils import(
    detrend,
    set_reference_time,
    prep_time_series
)

def slot_weighted_dcf(
    time_series_1,
    time_series_2,
    tau_limits,
    delta_tau,
    amplitudes_1=None,
    amplitudes_2=None,
    uncertainties_1=None,
    uncertainties_2=None
):
    '''This is a rewrite of the slot weighted DCF algorithm designed to work with data of a similar
    structure to the original Python implimentation, or alternatively using other common Python
    structures.

    :param time_series_1: list or numpy array of one or two dimensions representing the time series data.
        The first dimension represents the time series points and the second allows for amplitudes_1 and
        uncertainties_1 to be loaded as in the original pydcf implimentation. If one dimension, this
        only represents the observation times.
    :param time_series_2: list or numpy array of one or two dimensions representing the time series data.
        The first dimension represents the time series points and the second allows for amplitudes_1 and
        uncertainties_1 to be loaded as in the original pydcf implimentation. If one dimension, this
        only represents the observation times.
    :param tau_limits: list, tuple, or numpy array representing the minimum and maximum time lags to
        consider
    :param delta_tau: int or float representing the spacing between time lag bins. The bin width should
        be larger than the approximate spacing between data points
    :param amplitudes_1: list or numpy array representing the flux or magnitudes of the light curve at
        each point defined by time_series_1
    :param amplitudes_2: list or numpy array representing the flux or magnitudes of the light curve at
        each point defined by time_series_2
    :param uncertainties_1: list or numpy array representing the uncertainty in flux or magnitude of
        the light curve at each point defined by time_series_1
    :param uncertainties_2: list or numpy array representing the uncertainty in flux or magnitude of
        the light curve at each point defined by time_series_2
    :return: two instances of numpy arrays representing the discrete correlation coefficients and their
        uncertainties
    '''

    if time_series_1.ndim == 3:
        uncertainties_1 = time_series_1[:, -1]
    if time_series_1.ndim >= 2:
        amplitudes_1 = time_series_1[:, 1]
        time_series_1 = time_series_1[:, 0]
    if time_series_2.ndim == 3:
        uncertainties_2 = time_series_2[:, -1]
    if time_series_2.ndim >= 2:
        amplitudes_2 = time_series_2[:, 1]
        time_series_2 = time_series_2[:, 0]

    if amplitudes_1 is None:
        print("Please provide the amplitudes to compute the correlations")
        return 
    if amplitudes_2 is None:
        print("Please provide the amplitudes to compute the correlations")
        return 

    time_series_1, amplitudes_1, time_series_2, amplitudes_2 = prep_time_series(
        time_series_1,
        amplitudes_1,
        time_series_2,
        amplitudes_2,
        uncertainties_1=uncertainties_1,
        uncertainties_2=uncertainties_2
    )

    n_bins = int((np.max(tau_limits)-np.min(tau_limits)) / delta_tau + 0.5)
    
    tau_bins = np.linspace(
        np.min(tau_limits),
        np.max(tau_limits),
        n_bins
    )

    if uncertainties_1 is None:
        uncertainties_1 = np.zeros(np.shape(amplitudes_1))
    else:
        uncertainties_1 = np.asarray(uncertainties_1)
    if uncertainties_2 is None:
        uncertainties_2 = np.zeros(np.shape(amplitudes_2))
    else:
        uncertainties_2 = np.asarray(uncertainties_2)

    t1, t2 = np.meshgrid(time_series_1, time_series_2, indexing='ij')
    delta_times = t2 - t1

    output_dcf = np.empty((n_bins))
    output_dcf_err = np.empty((n_bins))
    
    for ii in range(n_bins):

        tau_index_mask_1, tau_index_mask_2 = np.where(
            (delta_times < (tau_bins[ii] + delta_tau/2)) & (delta_times > (tau_bins[ii] - delta_tau/2))
        )
        
        mean_amplitude_1 = np.mean(amplitudes_1[tau_index_mask_1])
        mean_amplitude_2 = np.mean(amplitudes_2[tau_index_mask_2])

        number_values = tau_index_mask_1.shape[0]

        dcf_denominator = np.sqrt(
            (np.var(amplitudes_1[tau_index_mask_1]) - np.mean(uncertainties_1[tau_index_mask_1])**2)
            * (np.var(amplitudes_2[tau_index_mask_2]) - np.mean(uncertainties_2[tau_index_mask_2])**2)
        )

        current_dcf = (
            amplitudes_1[tau_index_mask_1] - mean_amplitude_1
        ) * (
            amplitudes_2[tau_index_mask_2] - mean_amplitude_2
        ) / dcf_denominator

        output_dcf[ii] = np.sum(current_dcf) / number_values

        output_dcf_err[ii] = np.sqrt(np.sum((current_dcf - output_dcf[ii])**2)) / (number_values - 1)

    return output_dcf, output_dcf_err


def gaussian_kernel_dcf(
    time_series_1,
    time_series_2,
    tau_limits,
    delta_tau,
    amplitudes_1=None,
    amplitudes_2=None,
    uncertainties_1=None,
    uncertainties_2=None
):
    '''This is a rewrite of the slot weighted DCF algorithm designed to work with data of a similar
    structure to the original Python implimentation, or alternatively using other common Python
    structures.

    :param time_series_1: list or numpy array of one or two dimensions representing the time series data.
        The first dimension represents the time series points and the second allows for amplitudes_1 and
        uncertainties_1 to be loaded as in the original pydcf implimentation. If one dimension, this
        only represents the observation times.
    :param time_series_2: list or numpy array of one or two dimensions representing the time series data.
        The first dimension represents the time series points and the second allows for amplitudes_1 and
        uncertainties_1 to be loaded as in the original pydcf implimentation. If one dimension, this
        only represents the observation times.
    :param tau_limits: list, tuple, or numpy array representing the minimum and maximum time lags to
        consider
    :param delta_tau: int or float representing the spacing between time lag bins. The bin width should
        be larger than the approximate spacing between data points
    :param amplitudes_1: list or numpy array representing the flux or magnitudes of the light curve at
        each point defined by time_series_1
    :param amplitudes_2: list or numpy array representing the flux or magnitudes of the light curve at
        each point defined by time_series_2
    :param uncertainties_1: list or numpy array representing the uncertainty in flux or magnitude of
        the light curve at each point defined by time_series_1
    :param uncertainties_2: list or numpy array representing the uncertainty in flux or magnitude of
        the light curve at each point defined by time_series_2
    :return: two instances of numpy arrays representing the discrete correlation coefficients and their
        uncertainties
    '''

    if time_series_1.ndim == 3:
        uncertainties_1 = time_series_1[:, -1]
    if time_series_1.ndim >= 2:
        amplitudes_1 = time_series_1[:, 1]
        time_series_1 = time_series_1[:, 0]
    if time_series_2.ndim == 3:
        uncertainties_2 = time_series_2[:, -1]
    if time_series_2.ndim >= 2:
        amplitudes_2 = time_series_2[:, 1]
        time_series_2 = time_series_2[:, 0]

    if amplitudes_1 is None:
        print("Please provide the amplitudes to compute the correlations")
        return 
    if amplitudes_2 is None:
        print("Please provide the amplitudes to compute the correlations")
        return 

    gaussian_contribution = lambda x: np.exp(-x**2 / (2 * delta_tau**2)) / (2 * np.pi * delta_tau)**0.5
    contribution_threshold = gaussian_contribution(5 * delta_tau)

    time_series_1, amplitudes_1, time_series_2, amplitudes_2 = prep_time_series(
        time_series_1,
        amplitudes_1,
        time_series_2,
        amplitudes_2,
        uncertainties_1=uncertainties_1,
        uncertainties_2=uncertainties_2
    )

    n_bins = int((np.max(tau_limits)-np.min(tau_limits)) / delta_tau + 0.5)
    
    tau_bins = np.linspace(
        np.min(tau_limits),
        np.max(tau_limits),
        n_bins
    )

    if uncertainties_1 is None:
        uncertainties_1 = np.zeros(np.shape(amplitudes_1))
    else:
        uncertainties_1 = np.asarray(uncertainties_1)
    if uncertainties_2 is None:
        uncertainties_2 = np.zeros(np.shape(amplitudes_2))
    else:
        uncertainties_2 = np.asarray(uncertainties_2)

    t1, t2 = np.meshgrid(time_series_1, time_series_2, indexing='ij')
    delta_times = t2 - t1

    output_dcf = np.empty((n_bins))
    output_dcf_err = np.empty((n_bins))
    
    
    for ii in range(n_bins):

        gaussian_time_lag_distribution = gaussian_contribution(delta_times - tau_bins[ii])

        tau_index_mask_1, tau_index_mask_2 = np.where(
            (gaussian_time_lag_distribution >= contribution_threshold)
        )
        
        mean_amplitude_1 = np.mean(amplitudes_1[tau_index_mask_1] * gaussian_time_lag_distribution[tau_index_mask_1, tau_index_mask_2])
        mean_amplitude_2 = np.mean(amplitudes_2[tau_index_mask_2] * gaussian_time_lag_distribution[tau_index_mask_1, tau_index_mask_2])

        number_values = np.sum(gaussian_time_lag_distribution[tau_index_mask_1, tau_index_mask_2])

        dcf_denominator = np.sqrt(
            (np.var(amplitudes_1[tau_index_mask_1]) - np.mean(uncertainties_1[tau_index_mask_1])**2)
            * (np.var(amplitudes_2[tau_index_mask_2]) - np.mean(uncertainties_2[tau_index_mask_2])**2)
        )

        current_dcf = gaussian_time_lag_distribution[tau_index_mask_1, tau_index_mask_2] * (
            amplitudes_1[tau_index_mask_1] - mean_amplitude_1
        ) * (
            amplitudes_2[tau_index_mask_2] - mean_amplitude_2
        ) / dcf_denominator

        output_dcf[ii] = np.sum(current_dcf) / number_values

        output_dcf_err[ii] = np.sqrt(np.sum((current_dcf - output_dcf[ii])**2)) / (number_values - 1)

    return output_dcf, output_dcf_err








