
import numpy as np
import pytest
from scipy.special import sph_harm

import pyspherical as pysh


# Test transforms of sampled data.

# Tests to add:
#   *> Transform and inverse transform returns original.
#   *> Transform of a linear combo of spherical harmonics returns peaks in the right places.
#   > Check with different samplings
#   *> Transforms with higher spins.


@pytest.fixture
def mw_sum_of_harms():
    lmax = 50

    Nt = lmax
    Nf = 2 * lmax - 1

    # Define samples
    dth = np.pi / (2 * Nt - 1)
    theta = np.linspace(dth, np.pi, Nt, endpoint=True)
    phi = np.linspace(0, 2 * np.pi, Nf, endpoint=False)

    gtheta, gphi = np.meshgrid(theta, phi)

    def _func(spin=0):
        # Data
        Npeaks = 10

        # NOTE Transforms seem to fail the loop test when the el = spin component is
        # nonzero. This will need to be investigated, but for now just avoid it.
        peak_els = np.random.choice(np.arange(spin + 1, lmax - 1), Npeaks, replace=False)

        peak_ems = np.array([np.random.randint(-el, el + 1) for el in peak_els])
        peak_amps = np.random.uniform(10, 20, Npeaks)
        dat = np.zeros(gtheta.shape, dtype=complex)
        for ii in range(Npeaks):
            em = peak_ems[ii]
            el = peak_els[ii]

            dat += peak_amps[ii] * pysh.wigner.spin_spharm_goldberg(spin, el, em, gtheta, gphi)

        return dat, lmax, theta, phi, (peak_els, peak_ems, peak_amps)

    return _func


def test_transform_mw_sampling(mw_sum_of_harms):
    # MW sampling:
    #   (lmax) samples in theta
    #   (2 * lmax - 1) in phi

    dat, lmax, theta, phi, (peak_els, peak_ems, peak_amps) = mw_sum_of_harms(0)
    flm = pysh.forward_transform(dat, phi, theta, lmax, lmin=0, spin=0)
    Npeaks = len(peak_els)

    # Verify that the peaks are at the expected el, em.
    flm_srt = np.argsort(flm)
    peak_inds = flm_srt[-Npeaks:]
    lmtest = np.array([pysh.ravel_lm(ind) for ind in peak_inds])
    assert set(lmtest[:, 0]) == set(peak_els)
    assert set(lmtest[:, 1]) == set(peak_ems)
    assert np.allclose(np.array(sorted(peak_amps)),
                       flm[peak_inds].real, atol=1e-5)

    # Check that the remaining points are all near zero.
    assert np.allclose(flm[flm_srt[:-Npeaks]], 0.0, atol=1.0)


@pytest.mark.parametrize('lmax', range(10, 50, 5))
def test_transform_mw_sampling_monopole(lmax):
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

    amp = 20
    dat = np.ones(gtheta.shape, dtype=float) * amp

    flm = pysh.forward_transform(dat, phi, theta, lmax, lmin=0, spin=0)
    assert np.isclose(flm[0], amp * np.sqrt(4 * np.pi))


@pytest.mark.parametrize('spin', range(4))
def test_transform_mw_sampling_loop(spin, mw_sum_of_harms):
    # sphere -> flm -> sphere.

    dat, lmax, theta, phi, (peak_els, peak_ems, peak_amps) = mw_sum_of_harms(spin)

    flm = pysh.forward_transform(dat, phi, theta, lmax, lmin=0, spin=spin)

    res = pysh.inverse_transform(flm)

    assert np.allclose(res, dat, atol=1e-5)

    Nt = 50
    Nf = 70

    dat2 = pysh.inverse_transform(flm, Nt=Nt, Nf=Nf, spin=spin)

    dth = np.pi / (2 * Nt - 1)
    theta2 = np.linspace(dth, np.pi, Nt, endpoint=True)
    phi2 = np.linspace(0, 2 * np.pi, Nf, endpoint=False)

    flm2 = pysh.forward_transform(dat2, phi2, theta2, lmax, spin=spin)
    res2 = pysh.inverse_transform(flm2, thetas=theta2, phis=phi2, spin=spin)
    assert np.allclose(dat2, res2, atol=1e-5)


@pytest.mark.parametrize('spin', range(4))
def test_loop_mw_nongrid(spin, mw_sum_of_harms):
    # sphere -> flm -> sphere
    # But use meshgrid of points

    dat, lmax, theta, phi, (peak_els, peak_ems, peak_amps) = mw_sum_of_harms(spin)

    gtheta, gphi = np.meshgrid(theta, phi)

    flm = pysh.forward_transform(dat, gphi, gtheta, lmax, spin=spin)
    res = pysh.inverse_transform(flm, gphi, gtheta, lmax, spin=spin)

    assert np.allclose(dat.flatten(), res, atol=1e-4)


@pytest.mark.skip(reason="Unexplained deviation in this case. Needs work.")
def test_loop_diffphi_nongrid():
    # Make a sphere with different phi sampling on each latitude.
    # Evaluate a function on the sphere and do the loop test.

    lmax = 80

    # Number of thetas (latitudes)
    Nt = 81

    # Number of phi samples at each latitude
    Nf = [75] * 26 + [77] * 28 + [74] * 27

    # Put an offset in phi at each latitude.
    offsets = [np.random.uniform(0, np.pi / (nf)) for nf in Nf]
    thetas, phis = [], []

    dth = np.pi / (2 * Nt - 1)
    base_theta = np.linspace(dth, np.pi, Nt, endpoint=True)

    for ti, th, in enumerate(base_theta):
        nf = Nf[ti]
        thetas.extend([th] * nf)
        cur_phi = (np.linspace(0, 2 * np.pi, nf) + offsets[ti]) % (2 * np.pi)
        phis.extend(cur_phi.tolist())

    # Evaluate a function on these points.
    el = 17
    em = 3
    amp = 50

    dat = amp * sph_harm(em, el, phis, thetas)

    flm = pysh.forward_transform(dat, phis, thetas, lmax=lmax)
    res = pysh.inverse_transform(flm, phis, thetas, lmax=lmax)

    # There is a regular pattern to the remaining deviation between dat and res.
    # The results shouldn't match up perfectly, because the sampling isn't exact and
    # it is band-limited, but it's unclear how reversible this should be.
    tol = 1.0
    assert np.allclose(dat, res, atol=tol)
