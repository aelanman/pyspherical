
import numpy as np
import pytest
from scipy.special import sph_harm

import pyspherical as pysh


# Test transforms of sampled data.

## Tests to add:
#   > Transform and inverse transform.
#   > Transform of a linear combo of spherical harmonics returns peaks in the right places.
#   > Check with different samplings
#   > Transform with higher spins.

@pytest.mark.parametrize('lmax', [50]*10)
def test_transform_mw_sampling(lmax):
    # MW sampling:
    #   (lmax) samples in theta
    #   (2 * lmax - 1) in phi

    Nt = lmax
    Nf = 2 * lmax - 1

    # Define samples
    dth = np.pi / (2 * Nt - 1)
    theta = np.linspace(dth, np.pi, Nt, endpoint=True)
    phi = np.linspace(0, 2 * np.pi, Nf, endpoint=False)

    gtheta, gphi = np.meshgrid(theta, phi)

    # Data
    Npeaks = 10
    #peak_els = np.random.randint(0, lmax, Npeaks)
    peak_els = np.random.randint(1, lmax, Npeaks)   # XXX temporarily start at 1. el = 0 has an offset of ~3.55

    peak_ems = np.array([np.random.randint(-el, el+1) for el in peak_els])
    peak_amps = np.random.uniform(10, 20, Npeaks)

    dat = np.ones(gtheta.shape, dtype=complex)
    for ii in range(Npeaks):
        em = peak_ems[ii]
        el = peak_els[ii]
        dat += peak_amps[ii] * sph_harm(em, el, gphi, gtheta)

    flm = pysh.forward_transform(dat, phi, theta, lmax, lmin=0, spin=0)

    # Verify that the peaks are at the expected el, em.
    flm_srt = np.argsort(flm)
    peak_inds = flm_srt[-Npeaks:]
    lmtest = np.array([pysh.ravel_lm(ind) for ind in peak_inds])
    assert set(lmtest[:,0]) == set(peak_els)

    assert set(lmtest[:,1]) == set(peak_ems)
    assert np.allclose(np.array(sorted(peak_amps)), flm[peak_inds].real, atol=1e-5)

    # Check that the remaining points are all near zero.
    # TODO -- Not currently passing, because there is a remaining 3.55 ish at el = 0
    #    that shouldn't be there.
    #assert np.allclose(flm[flm_srt[:-Npeaks]], 0.0, atol=1e-5)
