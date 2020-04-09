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
        pysh.HarmonicFunction.current_dmat = None   # Reset current_dmat

    flm = np.zeros(lmax**2, dtype=complex)
    flm[pysh.utils.unravel_lm(el, em)] = amp

    test1 = pysh.inverse_transform(flm, thetas=theta, phis=phi, spin=spin)
    test2 = amp * \
        pysh.HarmonicFunction.spin_spherical_harmonic(
            el, em, spin, gtheta, gphi)
    test3 = amp * pysh.wigner.spin_spharm_goldberg(spin, el, em, gtheta, gphi)

    assert np.allclose(test1, test2, atol=1e-10)
    assert np.allclose(test2, test3, atol=1e-5)
    assert pysh.HarmonicFunction.current_dmat.lmax == lmax + 1


def test_wigner_symm():
    # Test symmetries of the Wigner-d function.
    # Appendix of Prezeau and Reinecke (2010)

    lmax = 50
    pysh.HarmonicFunction._set_wigner(lmax)

    def dl(el, m1, m2, theta):
        return pysh.HarmonicFunction.wigner_d(el, m1, m2, theta)

    Nth = 5
    for th in np.linspace(0, np.pi, Nth):
        for ll in range(15):
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
