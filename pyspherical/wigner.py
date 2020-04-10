"""Evaluation of Wigner-d functions and spin-weighted spherical harmonic functions."""

import numpy as np
import warnings
from numba import jit
from scipy.special import binom, factorial

from .utils import tri_ravel

__all__ = [
    'spin_spharm_goldberg',
    'DeltaMatrix',
    'spin_spherical_harmonic',
    'wigner_d',
    'get_cached_dmat',
    'clear_cached_dmat',
]


@jit(nopython=True)
def _dmat_eval(lmin, lmax, arr):
    # Evaluate the values of the Wigner-d matrices at pi/2.
    # arr = linear array, modified in-place
    arr[0] = 1.0
    for el in range(1, lmax + 1):
        dll0 = -np.sqrt((2 * el - 1) / float(2 * el)) * \
            arr[tri_ravel(el - 1, el - 1, 0)]
        arr[tri_ravel(el, el, 0)] = dll0
        for m2 in range(0, el + 1):
            if m2 > 0:
                dllm = np.sqrt(
                    (el / 2) * (2 * el - 1) / ((el + m2) * (el + m2 - 1))
                ) * arr[tri_ravel(el - 1, el - 1, m2 - 1)]
                arr[tri_ravel(el, el, m2)] = dllm
            for m1 in range(el - 1, m2 - 1, -1):
                fac1 = (2 * m2) / np.sqrt((el - m1) * (el + m1 + 1)) * \
                    arr[tri_ravel(el, m1 + 1, m2)]
                fac2 = 0.0
                if (m1 + 2) <= el:
                    fac2 = np.sqrt(
                        ((el - m1 - 1) * (el + m1 + 2))
                        / ((el - m1) * (el + m1 + 1))
                    ) * arr[tri_ravel(el, m1 + 2, m2)]
                arr[tri_ravel(el, m1, m2)] = fac1 - fac2


@jit(nopython=True)
def _access_element(l, m1, m2, arr):
    # Access stored elements, or use
    # symmetry relations for non-stored elements.

    _m1, _m2 = m1, m2

    fac = (-1)**(_m2 - _m1)     # For sign convention of the SSHT paper

    if _m1 < 0 and _m2 < 0:
        fac *= (-1)**(m1 - m2)
        m1 *= -1
        m2 *= -1
    elif _m1 < 0 and _m2 >= 0:
        fac *= (-1)**(l - _m2)
        m1 *= -1
    elif _m1 >= 0 and _m2 < 0:
        fac *= (-1)**(l + _m1)
        m2 *= -1

    # reverse if wrong order
    if m1 < m2:
        fac *= (-1)**(m1 - m2)
        m1, m2 = m2, m1

    val = fac * arr[tri_ravel(l, m1, m2)]
    if np.isnan(val):
        raise ValueError("Invalid entry in dmat")
    return val


class DeltaMatrix:
    """
    Wigner-d functions evaluated at pi/2.

    Only stores values for m1, m2 >= 0. Other values are returned by symmetry relations.

    Based on the methods in:
        S, Trapani, and Navaza J. Acta Crystallographica Section A, vol. 62, no. 4, 2006, pp. 262–69.


    Parameters
    ----------
    lmax: int
        Maximum multipole mode. (Optional)
        Defaults to sqrt(len(flm)).
    lmin: int
        Currently unused.
        Minimum multipole mode to compute. (Optional, default 0)
    """

    def __init__(self, lmax, lmin=0):
        arrsize = self.estimate_array_size(lmin, lmax)
        self._arr = np.empty(arrsize, dtype=np.float32)
        self.lmax = lmax
        self.lmin = lmin
        self.size = arrsize
        self._eval()

    def _eval(self):
        _dmat_eval(self.lmin, self.lmax, self._arr)

    @classmethod
    def estimate_array_size(cls, lmin, lmax):
        """Estimate size of the flattened array needed to store Delta_lmm values."""
        return (1 + lmax - lmin) * (6 + 5 * lmax + lmax**2 + 4
                                    * lmin + lmax * lmin + lmin**2) // 6

    def __getitem__(self, index):
        """
        Access stored elements, or use symmetry relations for non-stored elements.

        Parameters
        ----------
        index: tuple
            (l, m1, m2) delta matrix entry

        Returns
        -------
        float
            Value of Delta[el, m1, m2]
        """
        (l, m1, m2) = index
        return _access_element(l, m1, m2, self._arr)


@jit(nopython=True)
def _get_matrix_elements(el, m1, m2, arr, outarr):
    """
    Get an array of Delta[el, mp, m1] * Delta[el, mp, m2].

    Results are written into outarr
    """
    for mi, mp in enumerate(range(0, el + 1)):
        outarr[mi] = _access_element(
            el, mp, m1, arr) * _access_element(el, mp, m2, arr)


class HarmonicFunction:
    """
    Methods for calculating Wigner-d functions and spin-weighted spherical harmonics.

    Caches a :class:`DeltaMatrix` to speed up subsequent calculations.

    Not to be instantiated.
    """

    current_dmat = None

    def __init__(self):
        raise Exception("HarmonicFunction class is not instantiable.")

    @classmethod
    def _set_wigner(cls, lmax):
        if (cls.current_dmat is None) and (lmax is not None):
            cls.current_dmat = DeltaMatrix(lmax)
        if (cls.current_dmat is None) or (cls.current_dmat.lmax < lmax):
            cls.current_dmat = DeltaMatrix(lmax)

    @classmethod
    def wigner_d(cls, el, m1, m2, theta, lmax=None):
        theta = np.atleast_1d(theta)
        if lmax is None:
            lmax = el
        cls._set_wigner(lmax)

        mp = np.arange(1, el + 1)
        exp_fac = np.exp(1j * mp[None, :] * theta[..., None])
        dmats = np.empty(el + 1, dtype=float)
        _get_matrix_elements(el, m1, m2, cls.current_dmat._arr, dmats)

        val = (1j) ** (m2 - m1) * (
            np.sum((exp_fac + (-1.)**(m1 + m2 - 2 * el) / exp_fac)
                   * dmats[1:], axis=-1)
            + dmats[0]
        )
        if val.size == 1:
            return complex(val)
        return val.squeeze()

    @classmethod
    def spin_spherical_harmonic(cls, el, em, spin, theta, phi, lmax=None):
        theta = np.asarray(theta)
        phi = np.asarray(phi)

        if not theta.shape == phi.shape:
            raise ValueError("theta and phi must have the same shape.")

        return (-1.)**(em) * np.sqrt((2 * el + 1) / (4 * np.pi))\
            * np.exp(1j * em * phi) * cls.wigner_d(el, em, -1 * spin, theta, lmax=lmax)

    @classmethod
    def clear_dmat(cls):
        """Delete cached DeltaMatrix."""
        cls.current_dmat = None


def wigner_d(el, m1, m2, theta, lmax=None):
    """
    Evaluate the Wigner-d function (little-d) using cached values at pi/2.

    d^l_{m1, m2}(theta)

    Parameters
    ----------
    el, m1, m2: ints
        Indices of the d-function.
    theta: float or ndarray of float
        Angle argument(s) in radians.
    lmax: int
        Precompute the cached Delta matrix up to some maximum el.
        (Optional. Defaults to el or the maximum el used in the current Python session)

    Returns
    -------
    complex or ndarray of complex
        Value of Wigner-d function.
        If multiple theta values are given, multiple values are returned.

    Notes
    -----
    Uses eqn 8 of McEwan and Wiaux (2011), which cites:
        A. F. Nikiforov and V. B. Uvarov (1991)

    """
    return HarmonicFunction.wigner_d(el, m1, m2, theta, lmax)


def spin_spherical_harmonic(el, em, spin, theta, phi, lmax=None):
    """
    Evaluate the spin-weighted spherical harmonic.

    Obeys the standard numpy broadcasting for theta and phi.

    Parameters
    ----------
    el, em: ints
        Spherical harmonic mode.
    spin: int
        Spin of the function.
    theta: ndarray or float
        Colatitude(s) to evaluate to, in radians.
    phi: ndarray or float
        Azimuths to evaluate to, in radians.
    lmax: int
        Precompute the cached Delta matrix up to some maximum el.
        (Optional. Defaults to el or the maximum el used in the current Python session)

    Returns
    -------
    complex or ndarray of complex
        Values of the sYlm spin-weighted spherical harmonic function
        at spherical positions theta, phi.

    Notes
    -----
    Uses eqns (2) and (7) of McEwan and Wiaux (2011).
    If theta/phi are arrays, they must have the same shape.
    """
    return HarmonicFunction.spin_spherical_harmonic(el, em, spin, theta, phi, lmax)


def get_cached_dmat():
    """Return currently cached DeltaMatrix."""
    return HarmonicFunction.current_dmat


def clear_cached_dmat():
    """Delete cached DeltaMatrix."""
    HarmonicFunction.clear_dmat()


def _fac(val):
    return factorial(val, exact=True)


def spin_spharm_goldberg(spin, el, em, theta, phi):
    """
    Spin-s spherical harmonic function from Goldberg et al. (1967).

    Parameters
    ----------
    spin: int
        Spin of the function.
    el, em: ints
        Spherical harmonic mode.
    theta: array_like or float
        Colatitude(s) to evaluate to, in radians.
    phi: array_like or float
        Azimuths to evaluate to, in radians.

    Returns
    -------
    complex or ndarray of complex
        Values of the sYlm spin-weighted spherical harmonic function
        at spherical positions theta, phi.

    Notes
    -----
    If theta/phi are arrays, they must have the same shape.

    For nonzero spin, this function is unstable when theta/2 is close to a multiple of pi.
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    if not theta.shape == phi.shape:
        raise ValueError("theta and phi must have the same shape.")

    if (spin > 0) and (el < spin):
        return np.zeros_like(theta)

    term0 = (-1.)**em * np.sqrt(
        _fac(el + em) * _fac(el - em) * (2 * el + 1)
        / (4 * np.pi * _fac(el + spin) * _fac(el - spin))
    )
    term1 = np.sin(theta / 2)**(2 * el)

    # This will include divide by zeros. They are removed later.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='divide by zero encountered')
        warnings.filterwarnings('ignore', message='invalid value encountered in multiply')
        term2 = np.sum([
            binom(el - spin, r) * binom(el + spin, r + spin - em)
            * (-1)**(el - r - spin)
            * np.exp(1j * em * phi) * (1 / np.tan(theta / 2))**(2 * r + spin - em)
            for r in range(el - spin + 1)
        ], axis=0)

        res = term0 * term1 * term2
    res[np.isclose((theta / 2) % np.pi, 0)] = 0.0    # Singularities

    if res.size == 1:
        return complex(res)

    return res
