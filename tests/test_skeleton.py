import pytest
from diametery.skeleton import get_total_flux
import numpy as np

def test_get_total_flux():
    flux = np.array([
        [(0.2, 0.1), (0.3, 0.15), (-0.8, -0.2), (-0.4, 0.22)],
        [(0.4, -0.35), (0.6, 0.1), (0, 0), (-0.5, 0.03)],
        [(-0.5, 0.04), (0.9, -0.5), (0.7, -0.1), (-0.2, 0.2)]
    ])
    
    total_flux = np.array([
        [1, 4, 0, 0],
        [0, 0, 4, 0],
        [0, 0, 0, 2]
    ])
    a = get_total_flux(flux)
    assert np.all(a == total_flux)

def test_get_total_flux_cyclical():
    flux = np.array([
        [(0.2, 0.35), (0.3, 0.45)],
        [(0.4, -0.35), (-0.6, 0.1)]
    ])

    total_flux = np.array([
        [0, 1],
        [2, 1]
    ])
    a = get_total_flux(flux)
    assert np.all(a == total_flux)