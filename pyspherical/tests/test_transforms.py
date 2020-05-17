
import numpy as np
import pytest

import pyspherical as pysh


# Test transforms of sampled data.


@pytest.fixture
def mw_sum_of_harms():
    lmax = 10

    theta, phi = pysh.utils.get_grid_sampling(lmax)

    gtheta, gphi = np.meshgrid(theta, phi)

    def _func(spin=0, lmin=None):
        # Data
        Npeaks = 5

        if lmin is not None:
            lmin = max(spin, lmin) + 1
        else:
            lmin = spin + 1

        # NOTE Transforms seem to fail the loop test when the el = spin component is nonzero.
        peak_els = np.random.choice(np.arange(lmin, lmax - 1), Npeaks, replace=False)
        peak_ems = np.array([np.random.randint(-el, el + 1) for el in peak_els])
        peak_amps = np.random.uniform(10, 20, Npeaks)
        dat = np.zeros(gtheta.shape, dtype=complex)
        for ii in range(Npeaks):
            em = peak_ems[ii]
            el = peak_els[ii]
            dat += peak_amps[ii] * pysh.spin_spherical_harmonic(spin, el, em, gtheta, gphi)

        return dat, lmax, theta, phi, (peak_els, peak_ems, peak_amps)

    return _func


@pytest.mark.parametrize('spin', range(3))
def test_transform_mw_sampling(spin, mw_sum_of_harms):
    # MW sampling:
    #   (lmax) samples in theta
    #   (2 * lmax - 1) in phi

    dat, lmax, theta, phi, (peak_els, peak_ems, peak_amps) = mw_sum_of_harms(spin)
    flm = pysh.forward_transform(dat, phi, theta, lmax, lmin=0, spin=spin)
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


def test_transform_mw_loop_with_lmin(mw_sum_of_harms):
    # MW sampling:
    #   (lmax) samples in theta
    #   (2 * lmax - 1) in phi

    spin = 0
    lmin = 3
#    pysh.clear_cached_dmat()
    dat, lmax, theta, phi, (peak_els, peak_ems, peak_amps) = mw_sum_of_harms(spin, lmin=lmin)
    flm = pysh.forward_transform(dat, phi, theta, lmax, lmin=lmin, spin=spin)
    Npeaks = len(peak_els)

    # Verify that the peaks are at the expected el, em.
    flm_srt = np.argsort(flm)
    peak_inds = flm_srt[-Npeaks:]
    lmtest = np.array([pysh.ravel_lm(ind + lmin**2) for ind in peak_inds])
    assert set(lmtest[:, 0]) == set(peak_els)
    assert set(lmtest[:, 1]) == set(peak_ems)
    assert np.allclose(np.array(sorted(peak_amps)),
                       flm[peak_inds].real, atol=1e-5)

    # Check that the remaining points are all near zero.
    assert np.allclose(flm[flm_srt[:-Npeaks]], 0.0, atol=1.0)

    # Error with wrong flm shape
    with pytest.raises(ValueError, match='Invalid flm shape'):
        pysh.inverse_transform(flm, phi, theta, lmax, lmin=0, spin=spin)

    # Check that inverse works.
    res = pysh.inverse_transform(flm, phi, theta, lmax, lmin=lmin, spin=spin)

    flm2 = pysh.forward_transform(res, phi, theta, lmax, lmin=lmin, spin=spin)

    assert np.allclose(flm2, flm)
    assert np.allclose(res, dat)


@pytest.mark.parametrize('lmax', range(10, 50, 5))
def test_transform_mw_sampling_monopole(lmax):
    theta, phi = pysh.utils.get_grid_sampling(lmax)

    gtheta, gphi = np.meshgrid(theta, phi)

    amp = 20
    dat = np.ones(gtheta.shape, dtype=float) * amp

    flm = pysh.forward_transform(dat, phi, theta, lmax, lmin=0, spin=0)
    assert np.isclose(flm[0], amp * np.sqrt(4 * np.pi))


@pytest.mark.parametrize('spin', range(3))
def test_transform_mw_sampling_loop(spin, mw_sum_of_harms):
    # sphere -> flm -> sphere.

    dat, lmax, theta, phi, (peak_els, peak_ems, peak_amps) = mw_sum_of_harms(spin)

    flm = pysh.forward_transform(dat, phi, theta, lmax, lmin=0, spin=spin)

    res = pysh.inverse_transform(flm, spin=spin)

    # NOTE High tolerance is needed here.
    # flm and returned data match expectations from pyssht, but do not match the input data.
    # This seems to be highly-dependent on spin and numerical in origin.
    assert np.allclose(res, dat, atol=1)

    Nt = 50
    Nf = 70

    dat2 = pysh.inverse_transform(flm, Nt=Nt, Nf=Nf, spin=spin)

    dth = np.pi / (2 * Nt - 1)
    theta2 = np.linspace(dth, np.pi, Nt, endpoint=True)
    phi2 = np.linspace(0, 2 * np.pi, Nf, endpoint=False)

    flm2 = pysh.forward_transform(dat2, phi2, theta2, lmax, spin=spin)
    res2 = pysh.inverse_transform(flm2, thetas=theta2, phis=phi2, spin=spin)
    assert np.allclose(dat2, res2, atol=1e-4)


@pytest.mark.parametrize('spin', range(3))
def test_loop_mw_nongrid(spin, mw_sum_of_harms):
    # sphere -> flm -> sphere
    # But use meshgrid of points

    dat, lmax, theta, phi, (peak_els, peak_ems, peak_amps) = mw_sum_of_harms(spin)

    gtheta, gphi = np.meshgrid(theta, phi)

    flm = pysh.forward_transform(dat, gphi, gtheta, lmax, spin=spin)
    res = pysh.inverse_transform(flm, gphi, gtheta, lmax, spin=spin)

    assert np.allclose(dat.flatten(), res, atol=1)

# Slow
@pytest.mark.parametrize('offsets', [True, False])
@pytest.mark.parametrize('em', range(1, 15, 4))
def test_loop_diffphi_nongrid(offsets, em):
    # Make a sphere with different phi sampling on each latitude.
    # Evaluate a function on the sphere and do the loop test.

    lmax = 80

    # Number of thetas (latitudes)
    Nt = 81

    # Number of phi samples at each latitude
    Nfmean = 90
    Nf = [Nfmean - 1] * 26 + [Nfmean] * 28 + [Nfmean - 1] * 27

    # Put an offset in phi at each latitude.
    if offsets:
        offsets = [np.random.uniform(0, np.pi / (nf)) for nf in Nf]
    else:
        offsets = np.zeros(Nt)
    thetas, phis = [], []

    dth = np.pi / (2 * Nt - 1)
    base_theta = np.linspace(dth, np.pi, Nt, endpoint=True)

    for ti, th, in enumerate(base_theta):
        nf = Nf[ti]
        thetas.extend([th] * nf)
        cur_phi = (np.linspace(0, 2 * np.pi, nf, endpoint=False) + offsets[ti]) % (2 * np.pi)
        phis.extend(cur_phi.tolist())

    # Evaluate a function on these points.
    el = 17
    em = em
    amp = 50
    dat = amp * pysh.spin_spherical_harmonic(0, el, em, thetas, phis)

    flm = pysh.forward_transform(dat, phis, thetas, lmax=lmax)
    res = pysh.inverse_transform(flm, phis, thetas, lmax=lmax)

    # The results shouldn't match up perfectly, because the sampling isn't exact and
    # it is band-limited, but it's unclear how reversible this should be.
    # This is set with a very high tolerance due to error near the phi = 0.
    tol = 1.0
    assert np.allclose(dat, res, atol=tol)


@pytest.mark.parametrize('spin', range(3))
def test_loop_limited_mem(spin, mw_sum_of_harms):
    # Check that transforms still work with limited memory.

    dat, lmax, theta, phi, (peak_els, peak_ems, peak_amps) = mw_sum_of_harms(spin)

    # NOTE -- Falls into a recursive loop if lmax is too large -- Investigate
    lmax = 60

    flm = pysh.forward_transform(dat, phi, theta, lmax, spin=spin)
    res = pysh.inverse_transform(flm, phi, theta, lmax, spin=spin)
    flm2 = pysh.forward_transform(res, phi, theta, lmax, spin=spin)

    # Error if cached dmat exceeds limit:
    low_limit = 0.025  # MiB

    # Errors, because the current dmat would exceed this.
    with pytest.raises(ValueError, match="Cached DeltaMatrix exceeds"):
        pysh.set_cache_mem_limit(low_limit)

    pysh.clear_cached_dmat()
    # Limit cache memory to a small value.
    pysh.set_cache_mem_limit(low_limit)

    # Test lmin/lmax limiter.
    new_lmin, new_lmax = pysh.wigner.HarmonicFunction._limit_lmin_lmax(0, lmax, True)
    assert new_lmax == lmax and new_lmin > 0
    new_lmin, new_lmax = pysh.wigner.HarmonicFunction._limit_lmin_lmax(0, lmax, False)
    assert new_lmin == 0 and new_lmax < lmax

    # Error if trying to set lmax too high for memory limit.
    with pytest.raises(ValueError, match="Cannot construct DeltaMatrix within"):
        pysh.wigner.HarmonicFunction._set_wigner(0, 500, True)

    # Confirm that transformation still works both ways.
    flm3 = pysh.forward_transform(res, phi, theta, lmax, spin=spin)
    res2 = pysh.inverse_transform(flm3, phi, theta, lmax, spin=spin)

    details = pysh.get_cache_details()

    # Reset cache limit.
    pysh.set_cache_mem_limit(200)

    assert np.allclose(flm3, flm2)
    assert np.allclose(res, res2, atol=1e-4)
    assert details['size'] * np.zeros(1, dtype=np.float32).nbytes < details['cache_mem_limit']
