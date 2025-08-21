import numpy as np
from numpy.testing import assert_almost_equal
from modern_dcf.modern_dcf import (
    slot_weighted_dcf,
    gaussian_kernel_dcf
)


class TestModernDCF:

    def setup_method(self):
        my_time_series_1 = np.linspace(37, 801, 1001)
        my_time_series_2 = np.linspace(55, 800, 1201)
        my_seed = 17

        my_rng = np.random.default_rng(seed=my_seed)
        alt_rng = np.random.default_rng(seed=2*my_seed)

        my_data_1 = my_rng.normal(size=np.size(my_time_series_1))
        my_data_2 = np.interp(
            my_time_series_2, 
            my_time_series_1,
            my_data_1
        ) 
        my_data_2 += 0.05 * my_rng.normal(size=np.size(my_time_series_2))

        time_lag = 44

        my_data_2 = np.concatenate(
            (my_data_2[time_lag:], my_data_2[:time_lag])
        )

        my_uncertainties_1 = 0.021 * alt_rng.normal(size=np.size(my_time_series_1))
        my_uncertainties_2 = 0.025 * alt_rng.normal(size=np.size(my_time_series_2))
        
        combined_data_1 = np.asarray(
            [my_time_series_1, my_data_1, my_uncertainties_1]
        ).T 
        combined_data_2 = np.asarray(
            [my_time_series_2, my_data_2, my_uncertainties_2]
        ).T 

        part_data_1 = np.asarray(
            [my_time_series_1, my_data_1]
        ).T 
        part_data_2 = np.asarray(
            [my_time_series_2, my_data_2]
        ).T 
        
        
        self.time_data_1 = my_time_series_1
        self.time_data_2 = my_time_series_2
        self.amplitudes_1 = my_data_1
        self.amplitudes_2 = my_data_2
        self.uncertainties_1 = my_uncertainties_1
        self.uncertainties_2 = my_uncertainties_2
        self.full_data_1 = combined_data_1
        self.full_data_2 = combined_data_2
        self.partial_data_1 = part_data_1
        self.partial_data_2 = part_data_2

    def test_slot_dcf(self):

        tau_limits = [-75, 75]
        delta_tau = 3
        tau_bins = np.linspace(
            tau_limits[0], 
            tau_limits[1], 
            int((tau_limits[1]-tau_limits[0])/delta_tau+0.5)
        )
        
        dcf_1, dcf_err_1 = slot_weighted_dcf(
            self.time_data_1,
            self.time_data_2,
            tau_limits,
            delta_tau,
            amplitudes_1=self.amplitudes_1,
            amplitudes_2=self.amplitudes_2,
            uncertainties_1=self.uncertainties_1,
            uncertainties_2=self.uncertainties_2
        )

        dcf_p_1, dcf_err_p_1 = slot_weighted_dcf(
            self.time_data_1,
            self.time_data_2,
            tau_limits,
            delta_tau,
            amplitudes_1=self.amplitudes_1,
            amplitudes_2=self.amplitudes_2,
            uncertainties_1=None,
            uncertainties_2=None
        )

        t_dcf_1, t_dcf_err_1 = slot_weighted_dcf(
            self.full_data_1,
            self.full_data_2,
            tau_limits,
            delta_tau
        )

        t_dcf_p_1, t_dcf_err_p_1 = slot_weighted_dcf(
            self.partial_data_1,
            self.partial_data_2,
            tau_limits,
            delta_tau
        )

        # test the time lag is equal
        assert tau_bins[
            np.argmax(dcf_1)
        ] == tau_bins[
            np.argmax(t_dcf_1)
        ]
        
        # test the dcfs are almost equal
        for jj, value in enumerate(t_dcf_p_1):
            assert_almost_equal(dcf_p_1[jj], value)
        
        
    def test_gaussian_kernel_dcf(self):

        tau_limits = [-75, 75]
        delta_tau = 3
        tau_bins = np.linspace(
            tau_limits[0], 
            tau_limits[1], 
            int((tau_limits[1]-tau_limits[0])/delta_tau+0.5)
        )
        
        dcf_1, dcf_err_1 = gaussian_kernel_dcf(
            self.time_data_1,
            self.time_data_2,
            tau_limits,
            delta_tau,
            amplitudes_1=self.amplitudes_1,
            amplitudes_2=self.amplitudes_2,
            uncertainties_1=self.uncertainties_1,
            uncertainties_2=self.uncertainties_2
        )

        dcf_p_1, dcf_err_p_1 = gaussian_kernel_dcf(
            self.time_data_1,
            self.time_data_2,
            tau_limits,
            delta_tau,
            amplitudes_1=self.amplitudes_1,
            amplitudes_2=self.amplitudes_2,
            uncertainties_1=None,
            uncertainties_2=None
        )

        t_dcf_1, t_dcf_err_1 = gaussian_kernel_dcf(
            self.full_data_1,
            self.full_data_2,
            tau_limits,
            delta_tau
        )

        t_dcf_p_1, t_dcf_err_p_1 = gaussian_kernel_dcf(
            self.partial_data_1,
            self.partial_data_2,
            tau_limits,
            delta_tau
        )


        # test the time lag is equal
        assert tau_bins[
            np.argmax(dcf_1)
        ] == tau_bins[
            np.argmax(t_dcf_1)
        ]
        
        # test the dcfs are almost equal
        for jj, value in enumerate(t_dcf_p_1):
            assert_almost_equal(dcf_p_1[jj], value)
        
        

        