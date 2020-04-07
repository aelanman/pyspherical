
import numpy as np
import pytest
from scipy.special import sph_harm

import pyspherical as pysh


# Tests to add:
#   > Symmetry relations of delta matrices
#   > Symmetries of wigner-d functions at different angles
#   > Comparison against spherical_functions for l<32, if available
#   > Check that evaluated values are stored properly
#   > Values of wigner-d functions against mathematica-calculated values, using wolframclient
#   > Test value of spin_spherical_harmonic against scipy for spin=0.


class TestSphHarm:

    lmax = 5

    def setup(self):
        Nt = 701   # Number of samples in theta (must be odd)
        Nf = 401  # Samples in phi
        pysh.HarmonicFunction.current_dmat = None   # Reset current_dmat
        # Define samples
        dth = np.pi / (2 * Nt - 1)
        theta = np.linspace(dth, np.pi, Nt, endpoint=True)
        phi = np.linspace(0, 2 * np.pi, Nf, endpoint=False)

        # Data, shape (Nf , Nt)
        self.gtheta, self.gphi = np.meshgrid(theta, phi)

    @pytest.mark.parametrize('el, em', [(el, em) for el in range(0, lmax) for em in range(-el, el + 1)])
    def test_spherical_harmonic_spin0(self, el, em):
        # Check against scipy calculation
        # for spin 0

        dat = sph_harm(em, el, self.gphi, self.gtheta)

        # Scipy includes Condon-Shortley phase def.
        res = (-1)**em * pysh.HarmonicFunction.spin_spherical_harmonic(el,
                                                                       em, 0, self.gtheta, self.gphi, lmax=self.lmax)

        assert pysh.HarmonicFunction.current_dmat.lmax == self.lmax
        assert np.allclose(dat, res, atol=1e-4)
