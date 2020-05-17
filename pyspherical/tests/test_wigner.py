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

    # Error cases
    with pytest.raises(ValueError, match="theta and phi must have"):
        pysh.spin_spherical_harmonic(spin, el, em, theta, gphi)

    with pytest.raises(ValueError, match="theta and phi must have"):
        pysh.spin_spharm_goldberg(spin, el, em, theta, gphi)

    test1 = pysh.inverse_transform(flm, thetas=theta, phis=phi, spin=spin)
    test2 = amp * \
        pysh.spin_spherical_harmonic(
            spin, el, em, gtheta, gphi, lmax=lmax)
    test3 = amp * pysh.wigner.spin_spharm_goldberg(spin, el, em, gtheta, gphi)

    assert np.allclose(test1, test2, atol=1e-10)
    assert np.allclose(test2, test3, atol=1e-5)
    assert pysh.get_cached_dmat().lmax == lmax


def test_wigner_symm():
    # Test symmetries of the Wigner-d function.
    # Appendix of Prezeau and Reinecke (2010)

    # Test error case:
    with pytest.raises(ValueError, match='el 3 is less than lmin'):
        pysh.wigner_d(3, 0, 0, 0.5, lmin=5)

    lmax = 15

    def dl(el, m1, m2, theta):
        return pysh.wigner_d(el, m1, m2, theta, lmax=lmax)

    Nth = 5
    for th in np.linspace(0, np.pi, Nth):
        for ll in range(lmax):
            for m in range(-ll, ll + 1):
                for mm in range(-ll, ll + 1):
                    val1 = dl(ll, m, mm, th)
                    assert np.isclose(val1, (-1)**(m - mm) * dl(ll, -m, -mm, th))
                    assert np.isclose(val1, (-1)**(m - mm) * dl(ll, mm, m, th))
                    assert np.isclose(val1, dl(ll, -mm, -m, th))
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

    # Check error cases:
    with pytest.raises(ValueError, match="l < lmin. Need to re-evaluate"):
        dmat2[3, 0, 1]
    with pytest.raises(ValueError, match="l > lmax. Need to re-evaluate"):
        dmat2[22, 0, 1]


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
            _get_array_params(None, lmax, arrsize, strict=True)
        with pytest.raises(ValueError, match="Invalid"):
            _get_array_params(lmin, None, arrsize, strict=True)


@pytest.mark.parametrize(
    ('lmin', 'lmax', 'arrsize'),
    [
        (0, 15, 400),
        (4, 9, 70),
        (4, 9, 157)
    ]
)
def test_dmat_params_nonstrict(lmin, lmax, arrsize):
    # Ensure the returned size is smaller than the limits for various lmin/lmax.
    _get_array_params = pysh.DeltaMatrix._get_array_params

    # Choose a new lmin:
    lmin_new, _, arrsize_new = _get_array_params(lmax=lmax, arrsize=arrsize)
    assert (lmin_new >= lmin) and (arrsize_new <= arrsize)

    # Choose a new lmax:
    _, lmax_new, arrsize_new = _get_array_params(lmin=lmin, arrsize=arrsize)
    assert (lmax_new <= lmax) and (arrsize_new <= arrsize)


# Slow
def test_dmat_largeL():
    # Check that DeltaMatrix elements are accurate for some large el,
    # compared with the precise results of sympy.
    # This test may be slow.

    pytest.importorskip('sympy')
    from sympy.physics.quantum.spin import Rotation
    el, m, mp = 200, 77, 59

    wigd = Rotation.d(el, m, mp, np.pi / 2)
    val = complex(wigd.doit())

    pysh.wigner.HarmonicFunction._set_wigner(180, 210)

    assert np.isclose(val.real, pysh.wigner_d(el, m, mp, np.pi / 2), atol=1e-6)


# Slow
def test_dmat_64bit():
    # Check the DeltaMatrix calculated with double precision.
    # TODO Add option to run through the full range of m/mp, when allowing slow tests.
    pytest.importorskip('sympy')
    from sympy.physics.quantum.spin import Rotation
    el = 200

    pysh.wigner.HarmonicFunction._set_wigner(el - 1, el + 1, dtype=np.float64)

    # Choose random m, mp.
    for m, mp in np.random.randint(0, el + 1, (1, 2)):
        wigd = Rotation.d(el, m, mp, np.pi / 2)
        val = complex(wigd.doit())

        # Different sign convention.
        assert np.isclose(val.real, (-1)**(m + mp)
                          * pysh.wigner_d(el, m, mp, np.pi / 2), atol=1e-12)


def test_error():
    with pytest.raises(Exception, match="HarmonicFunction class is not"):
        pysh.wigner.HarmonicFunction()
