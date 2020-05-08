import numpy as np
import pytest

import pyspherical as pysh


@pytest.fixture
def mw_sampling():
    # MW sampling of a sphere
    Nt = 701   # Number of samples in theta (must be odd)
    Nf = 401  # Samples in phi
    theta, phi = pysh.utils.get_grid_sampling(Nt=Nt, Nf=Nf)
    gtheta, gphi = np.meshgrid(theta, phi)
    return theta, phi, gtheta, gphi


@pytest.mark.parametrize('slm', ((spin, el, em)
                                 for spin in range(3)
                                 for el in range(spin, 5)
                                 for em in range(-el, el + 1)
                                 )
                         )
def test_transform_eval_compare(mw_sampling, slm):
    # Compare evaluation of the spherical harmonic to the result returned
    # by the inverse transform.
    # Also compare spin_spherical_harmonic to goldberg eval.
    amp = 20
    lmax = 5
    spin, el, em = slm

    theta, phi, gtheta, gphi = mw_sampling

    if (spin, el, em) == (0, 0, 0):
        pysh.clear_cached_dmat()   # Reset current_dmat
        assert pysh.get_cached_dmat() is None

    flm = np.zeros(lmax**2, dtype=complex)
    flm[pysh.utils.unravel_lm(el, em)] = amp

    test1 = pysh.inverse_transform(flm, thetas=theta, phis=phi, spin=spin)
    test2 = amp * \
        pysh.spin_spherical_harmonic(
            el, em, spin, gtheta, gphi)
    test3 = amp * pysh.wigner.spin_spharm_goldberg(spin, el, em, gtheta, gphi)

    assert np.allclose(test1, test2, atol=1e-10)
    assert np.allclose(test2, test3, atol=1e-5)
    assert pysh.get_cached_dmat().lmax == lmax + 1


def test_wigner_symm():
    # Test symmetries of the Wigner-d function.
    # Appendix of Prezeau and Reinecke (2010)

    lmax = 15

    def dl(el, m1, m2, theta):
        return pysh.wigner_d(el, m1, m2, theta, lmax)

    Nth = 5
    for th in np.linspace(0, np.pi, Nth):
        for ll in range(lmax):
            for m in range(-ll, ll + 1):
                for mm in range(-ll, ll + 1):
                    val1 = dl(ll, m, mm, th)
                    assert val1 == (-1)**(m - mm) * dl(ll, -m, -mm, th)
                    assert val1 == (-1)**(m - mm) * dl(ll, mm, m, th)
                    assert val1 == dl(ll, -mm, -m, th)
                    assert np.isclose(dl(ll, m, mm, -th),
                                      dl(ll, mm, m, th), atol=1e-10)
                    assert np.isclose(
                        dl(ll, m, mm, np.pi - th), (-1)**(ll - mm) * dl(ll, -m, mm, th), atol=1e-10)


def test_delta_matrix_inits():
    # Initializing delta matrices with different conditions.

    lmax = 20

    # Full:
    dmat1 = pysh.DeltaMatrix(lmax)

    # With lmin, but no starting array.
    dmat2 = pysh.DeltaMatrix(lmax, lmin=5)

    # Subset of existing arr0:
    dmat3 = pysh.DeltaMatrix(15, lmin=6, dmat=dmat1)

    # Starting from end of previous.
    dmat4 = pysh.DeltaMatrix(lmax, lmin=15, dmat=dmat3)

    # Starting from after end of previous.
    dmat5 = pysh.DeltaMatrix(20, lmin=9, dmat=pysh.DeltaMatrix(7))

    # Given an existing dmat, will it be copied correctly?
    dmat6 = pysh.DeltaMatrix(lmax, lmin=0, dmat=dmat1)

    # For each, check that the results match with the full.

    for dm in [dmat2, dmat3, dmat4, dmat5, dmat6]:
        ln, lx = dm.lmin, dm.lmax
        for el in range(ln, lx):
            for m1 in range(-el, el + 1):
                for m2 in range(-el, el + 1):
                    assert dm[el, m1, m2] == dmat1[el, m1, m2]


@pytest.mark.parametrize(
        ('err', 'lmin', 'lmax', 'arrsize'),
            [
                (False, 0, 15, 816),
                (False, 4, 9, 200),
                (True, 4, 9, 157)
            ]
        )
def test_dmat_params(err, lmin, lmax, arrsize):
    # Getting missing parameter from the set lmin, lmax, arrsize
    _get_array_params = pysh.DeltaMatrix._get_array_params
    if not err:
        assert _get_array_params(None, lmax, arrsize) == (lmin, lmax, arrsize)
        assert _get_array_params(lmin, None, arrsize) == (lmin, lmax, arrsize)
        assert _get_array_params(lmin, lmax, None) == (lmin, lmax, arrsize)
    else:
        with pytest.raises(ValueError, match="Invalid"):
            _get_array_params(None, lmax, arrsize)
        with pytest.raises(ValueError, match="Invalid"):
            _get_array_params(lmin, None, arrsize)
