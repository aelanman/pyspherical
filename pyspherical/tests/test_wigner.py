import numpy as np
import pytest
from scipy.special import sph_harm, binom, factorial

import pyspherical as pysh


@pytest.fixture
def mw_sampling():
    # MW sampling of a sphere
    Nt = 701   # Number of samples in theta (must be odd)
    Nf = 401  # Samples in phi

    # Define samples
    dth = np.pi / (2 * Nt - 1)
    theta = np.linspace(dth, np.pi, Nt, endpoint=True)
    phi = np.linspace(0, 2 * np.pi, Nf, endpoint=False)

    return theta, phi


def fac(val):
    return factorial(val, exact=True)


def spin_spharm_goldberg(spin, el, em, theta, phi):
    # Spin-S spherical harmonic function from Goldberg et al. (1967)

    term0 = (-1)**em * np.sqrt(
        fac(el + em) * fac(el - em) * (2 * el + 1)
        / (4 * np.pi * fac(el + spin) * fac(el - spin))
    )
    term1 = np.sin(theta / 2)**(2 * el)
    term2 = np.sum(
        np.fromiter((
            binom(el - spin, r) * binom(el + spin, r
                                        + spin - em) * (-1)**(el - r - spin)
            * np.exp(1j * em * phi) * (1 / np.tan(theta / 2))**(2 * r + spin - em)
            for r in range(el - spin + 1)
        ), dtype=complex),
    )
    return term0 * term1 * term2


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

        res = pysh.HarmonicFunction.spin_spherical_harmonic(
            el, em, 0, self.gtheta, self.gphi, lmax=self.lmax
        )

        assert pysh.HarmonicFunction.current_dmat.lmax == self.lmax
        assert np.allclose(dat, res, atol=1e-4)


@pytest.mark.parametrize('slm', ((spin, el, em)
                                 for spin in range(3)
                                 for el in range(spin, 5)
                                 for em in range(-el, el + 1)
                                 )
                         )
def test_spherical_harmonic_nonzero_spin(slm, mw_sampling):
    spin, el, em = slm
    theta, phi = mw_sampling
    pysh.HarmonicFunction._set_wigner(5)
    for th in theta[::50]:
        for fi in phi[::50]:
            res = pysh.HarmonicFunction.spin_spherical_harmonic(
                el, em, spin, th, fi
            )
            comp = spin_spharm_goldberg(spin, el, em, th, fi)

            assert np.isclose(res, comp, atol=1e-5)


@pytest.mark.parametrize('slm', ((spin, el, em)
                                 for spin in range(3)
                                 for el in range(spin, 5)
                                 for em in range(-el, el + 1)
                                 )
                         )
def test_transform_eval_compare(slm):
    # Compare evaluation of the spherical harmonic to the result returned
    # by the inverse transform.
    amp = 20
    lmax = 5
    spin, el, em = slm

    flm = np.zeros(lmax**2, dtype=complex)
    flm[pysh.utils.unravel_lm(el, em)] = amp
    theta, phi = pysh.utils.get_grid_sampling(lmax)
    gtheta, gphi = np.meshgrid(theta, phi)
    test1 = pysh.inverse_transform(flm, thetas=theta, phis=phi, spin=spin)
    test2 = amp * \
        pysh.HarmonicFunction.spin_spherical_harmonic(
            el, em, spin, gtheta, gphi)
    test3 = amp * np.array([[spin_spharm_goldberg(spin, el, em, th, fi)
                             for th in theta] for fi in phi])

    assert np.allclose(test1, test2, atol=1e-10)
    assert np.allclose(test2, test3, atol=1e-5)


def test_wigner_symm():
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
