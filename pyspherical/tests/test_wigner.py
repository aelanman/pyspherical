
import numpy as np
import pytest
from scipy.special import sph_harm

import pyspherical as pysh


## Tests to add:
#   > Symmetry relations of delta matrices
#   > Symmetries of wigner-d functions at different angles
#   > Comparison against spherical_functions for l<32, if available
#   > Check that evaluated values are stored properly
#   > Values of wigner-d functions against mathematica-calculated values, using wolframclient
#   > Test value of spin_spherical_harmonic against scipy for spin=0.


# TODO -- Parametrize to use multiple el/em. Move sampling setup out and make a class.
def test_spherical_harmonic_spin0():
    # Check against scipy calculation
    Nt = 701   # Number of samples in theta (must be odd)
    Nf = 401  # Samples in phi
    
    # Define samples
    dth = np.pi/(2*Nt-1)
    theta = np.linspace(dth, np.pi, Nt, endpoint=True)
    phi = np.linspace(0, 2*np.pi, Nf, endpoint=False)
    
    # Data, shape (Nf , Nt)
    gtheta, gphi = np.meshgrid(theta, phi)
    dat = sph_harm(3,5, gphi, gtheta)


    res = pysh.spin_spherical_harmonic(5, 3, 0, gtheta, gphi)

    # Seems to be off in phase...

    import IPython; IPython.embed()i

test_spherical_harmonic_spin0()
    
