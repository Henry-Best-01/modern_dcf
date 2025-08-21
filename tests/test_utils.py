import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from src.modern_dcf.utils import (
    detrend,
    set_reference_time,
    prep_time_series
)

"""Tests for `modern_dcf` package."""
class TestUtils:

    def setup_method(self):
        first_time_series = [1, 3, 9, 17, 32]
        second_time_series = np.asarray([-20, -17, -4, 0])
        first_amplitude_series = [20, 18, 18.3, 18, 17]
        second_amplitude_series = np.asarray([0.2, 0.4, 0.1, 0.2])
        first_errors = [1, 1, 1, 1, 1]
        second_errors = None

        self.first_times = first_time_series
        self.second_times = second_time_series
        self.first_amplitudes = first_amplitude_series
        self.second_amplitudes = second_amplitude_series
        self.first_uncertainties = first_errors
        self.second_uncertainties = second_errors


    def test_detrend(self):
        t1 = self.first_times.copy()
        t2 = self.second_times.copy()
        a1 = self.first_amplitudes.copy()
        a2 = self.second_amplitudes.copy()
        e1 = self.first_uncertainties.copy()
        if self.second_uncertainties is not None:
            e2 = self.second_uncertainties.copy()
        else: 
            e2 = None

        assert np.mean(a1) != 0
        assert np.var(a1) != 0

        assert np.mean(a2) != 0
        assert np.var(a2) != 0
        
        assert not isinstance(self.first_amplitudes, np.ndarray)
        assert isinstance(self.second_amplitudes, np.ndarray)

        new_a1 = detrend(t1, a1, uncertainties=e1)
        new_a2 = detrend(t2, a2, e2)

        assert_almost_equal(np.mean(new_a1), 0)
        assert np.var(new_a1) != 0
        
        assert_almost_equal(np.mean(new_a2), 0)
        assert np.var(new_a2) != 0

        with pytest.raises(ValueError):
            detrend(t1, a2)

    
    def test_set_reference_time(self):
        t1 = self.first_times.copy()
        t2 = self.second_times.copy()

        assert isinstance(t1, list)
        assert isinstance(t2, np.ndarray)

        total_min = np.min(
            np.concatenate((t1, t2))
        )

        new_t1, new_t2 = set_reference_time(t1, t2)

        assert np.min(
            np.concatenate((new_t1, new_t2))
        ) == 0


        assert np.sum(
            np.asarray(self.first_times) - new_t1 - total_min
        ) == 0

        assert np.sum(
            np.asarray(self.second_times) - new_t2 - total_min
        ) == 0
        
    def test_prep_time_series(self):
        t1 = self.first_times.copy()
        t2 = self.second_times.copy()
        a1 = self.first_amplitudes.copy()
        a2 = self.second_amplitudes.copy()
        e1 = self.first_uncertainties.copy()
        if self.second_uncertainties is not None:
            e2 = self.second_uncertainties.copy()
        else: 
            e2 = None
        

        total_min = np.min(
            [np.min(t2), np.min(t1)]
        )

        new_t1, new_a1, new_t2, new_a2 = prep_time_series(
            t1, 
            a1, 
            t2, 
            a2, 
            uncertainties_1=e1, 
            uncertainties_2=e2
        )

        assert isinstance(new_a1, np.ndarray)
        assert isinstance(new_a2, np.ndarray)
        assert isinstance(new_t1, np.ndarray)
        assert isinstance(new_t2, np.ndarray)

        for jj in range(len(t1)):
            assert self.first_times[jj] - new_t1[jj] == total_min
        for jj in range(len(t2)):
            assert self.second_times[jj] - new_t2[jj] == total_min
        
                
        

    


