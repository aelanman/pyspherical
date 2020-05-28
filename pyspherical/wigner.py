"""Evaluation of Wigner-d functions and spin-weighted spherical harmonic functions."""

import numpy as np
import warnings
from numba import jit
from scipy.special import binom, factorial

from .utils import tri_ravel, tri_base, el_block_size


__all__ = [
    'spin_spharm_goldberg',
    'DeltaMatrix',
    'spin_spherical_harmonic',
    'wigner_d',
    'get_cached_dmat',
    'clear_cached_dmat',
    'set_cache_mem_limit',
    'get_cache_details',
]


@jit(nopython=True)
def _dmat_eval(lmax, arr, lmin=0, lstart=None, arr0=None):
    # Evaluate the values of the Wigner-d matrices at pi/2.
    # arr = linear array, modified in-place
    if arr0 is not None:
        arr[:len(arr0)] = arr0
    else:
        arr[0] = 1.0
    if lstart is None:
        lstart = lmin

    offset = tri_ravel(lmin, lmin, 0)
    for el in range(lstart + 1, lmax + 1):
        if el <= lmin + 1:
            # Shift previous result back.
            elm2_size = el_block_size(el - 2)
            # If this is the first step, the el-1 block is
            # in the zeroth position.
            if el == lstart + 1:
                elm2_size = 0
            elm1_size = el_block_size(el - 1)
            arr[:elm1_size] = arr[elm2_size:elm2_size + elm1_size].copy()
            offset = tri_base(el - 1)
        dll0 = -np.sqrt((2 * el - 1) / float(2 * el)) * \
            arr[tri_ravel(el - 1, el - 1, 0) - offset]
        arr[tri_ravel(el, el, 0) - offset] = dll0
        for m2 in range(0, el + 1):
            if m2 > 0:
                dllm = np.sqrt(
                    (el / 2) * (2 * el - 1) / ((el + m2) * (el + m2 - 1))
                ) * arr[tri_ravel(el - 1, el - 1, m2 - 1) - offset]
                arr[tri_ravel(el, el, m2) - offset] = dllm
            for m1 in range(el - 1, m2 - 1, -1):
                fac1 = (2 * m2) / np.sqrt((el - m1) * (el + m1 + 1)) * \
                    arr[tri_ravel(el, m1 + 1, m2) - offset]
                fac2 = 0.0
                if (m1 + 2) <= el:
                    fac2 = np.sqrt(
                        ((el - m1 - 1) * (el + m1 + 2))
                        / ((el - m1) * (el + m1 + 1))
                    ) * arr[tri_ravel(el, m1 + 2, m2) - offset]
                arr[tri_ravel(el, m1, m2) - offset] = fac1 - fac2


@jit(nopython=True)
def _access_element(l, m1, m2, arr, lmin=0):
    # Access stored elements, or use
    # symmetry relations for non-stored elements.

    _m1, _m2 = m1, m2

    fac = (-1.)**(_m2 - _m1)     # For sign convention of the SSHT paper

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
        fac *= (-1.)**(m1 - m2)
        m1, m2 = m2, m1

    val = fac * arr[tri_ravel(l, m1, m2) - tri_ravel(lmin, lmin, 0)]
    if np.isnan(val):
        raise ValueError("Invalid entry in dmat")
    return val


class DeltaMatrix:
    """
    Wigner-d functions evaluated at pi/2.

    Only stores values for m1, m2 >= 0. Other values are returned by symmetry relations.

    Based on the methods in:
        S, Trapani, and Navaza J. Acta Crystallographica Section A,
        vol. 62, no. 4, 2006, pp. 262–69.


    Parameters
    ----------
    lmax: int
        Maximum multipole mode. (Optional)
        Defaults to sqrt(len(flm)).
    lmin: int
        Minimum multipole mode to compute. (Optional, default 0)
    dmat: DeltaMatrix
        Initialize using the data in another DeltaMatrix to start.
        This will speed up computation for larger el, if dmat.lmax <= lmax.
    dtype: type
        Provide a dtype for the array.
        This defaults to 32-bit floats, which will be good to about four decimal
        places.
    """

    def __init__(self, lmax, lmin=0, dmat=None, dtype=np.float32):
        arrsize = self.estimate_array_size(lmin, lmax)
        self._arr = np.empty(arrsize, dtype=dtype)
        self.dtype = dtype
        self.lmax = lmax
        self.lmin = lmin
        self.size = arrsize
        self._eval(dmat)

    def _eval(self, old_dmat=None):
        arr0 = None
        lstart = 0
        if old_dmat is not None:
            # Start using data from old matrix.
            oln, olx = old_dmat.lmin, old_dmat.lmax
            ln, lx = self.lmin, self.lmax
            # Case 0:  [ ( ) ]
            if (oln <= ln) and (lx <= olx):
                arr0 = old_dmat._arr[tri_base(ln) - tri_base(oln):tri_base(lx + 1) - tri_base(oln)]
                lstart = lx
            # Case 1:  [ ( ] )
            elif (oln <= ln <= olx) and (lx > olx):
                arr0 = old_dmat._arr[tri_base(ln) - tri_base(oln):tri_base(olx + 1) - tri_base(oln)]
                lstart = olx
            # Case 2:  [ ] ( )
            elif (olx < ln):
                base = tri_base(olx) - tri_base(oln)
                arr0 = old_dmat._arr[base: base + el_block_size(olx)]
                lstart = olx    # Start from the topmost block.
        _dmat_eval(self.lmax, self._arr, lstart=lstart, arr0=arr0, lmin=self.lmin)

    @classmethod
    def estimate_array_size(cls, lmin, lmax):
        """Estimate size of the flattened array needed to store Delta_lmm values."""
        return (1 + lmax - lmin) * (6 + 5 * lmax + lmax**2 + 4
                                    * lmin + lmax * lmin + lmin**2) // 6

    @classmethod
    def _get_array_params(cls, lmin=None, lmax=None, arrsize=None, strict=False):
        # Fill in the missing parameter.
        # Only one input may be None at a time!

        if arrsize is None:
            arrsize = cls.estimate_array_size(lmin, lmax)

        if lmax is None:
            lmax = lmin
            while True:
                s = cls.estimate_array_size(lmin, lmax)
                if s > arrsize and strict:
                    raise ValueError("Invalid combination.")
                if s > arrsize:
                    lmax -= 1
                    break
                elif s == arrsize:
                    break
                lmax += 1

        if lmin is None:
            lmin = lmax
            while True:
                s = cls.estimate_array_size(lmin, lmax)
                if s > arrsize and strict:
                    raise ValueError("Invalid combination.")
                if s == arrsize:
                    break
                if s > arrsize:
                    lmin += 1
                    break
                lmin -= 1
        arrsize = cls.estimate_array_size(lmin, lmax)
        return (lmin, lmax, arrsize)

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
        if l < self.lmin:
            raise ValueError("l < lmin. Need to re-evaluate delta matrix.")
        if l > self.lmax:
            raise ValueError("l > lmax. Need to re-evaluate delta matrix.")
        return _access_element(l, m1, m2, self._arr, self.lmin)


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
    cache_mem_limit = 500 * 2**20   # 500 MiB, lmax ~ 921 for lmin = 0

    # This is the maximum el mode that whose full m1/m2 block can be
    # contained in the allowed memory limit.
    # Reset by set_cache_mem_limit()
    maximum_el = 10238

    def __init__(self):
        raise Exception("HarmonicFunction class is not instantiable.")

    @classmethod
    def set_cache_mem_limit(cls, maxmem):
        newlim = maxmem * 2**20
        if cls.current_dmat is not None and cls.current_dmat._arr.nbytes > newlim:
            raise ValueError(f"Cached DeltaMatrix exceeds new limit {maxmem} MiB."
                             " Clear it using clear_cached_dmat() first.")
        cls.cache_mem_limit = newlim
        max_block_size = cls._est_arrsize_limit(cls.cache_mem_limit)
        cls.maximum_el = np.floor((np.sqrt(1 + 8 * max_block_size) - 3) / 2)

    @classmethod
    def get_cache_details(cls):
        return_dict = {
            'cache_mem_limit': cls.cache_mem_limit,
            'maximum_el': cls.maximum_el
        }
        if cls.current_dmat is not None:
            return_dict.update({
                'lmin': cls.current_dmat.lmin,
                'lmax': cls.current_dmat.lmax,
                'size': cls.current_dmat.size,
            })
        return return_dict

    @classmethod
    def _est_arrsize_limit(cls, maxmem, double=False):
        # maxmem = memory limit in bytes
        dtype = np.float32
        if double:
            dtype = np.float64
        return int(np.floor(maxmem / np.zeros(1, dtype=dtype).nbytes))

    @classmethod
    def _limit_lmin_lmax(cls, lmin, lmax, high, dtype=None):
        """
        Choose new lmin/lmax respecting array size limits.

        Parameters
        ----------
        lmin, lmax: int
            Desired lmin/lmax.
        high: bool
            If the given limits are not possible under memory
            restrictions, this chooses whether the returned range
            should include lmin (False) or lmax (True).
        """
        if dtype is np.float64:
            double = True
        else:
            double = False
        mem_lim = getattr(cls, 'cache_mem_limit')
        max_arrsize = cls._est_arrsize_limit(mem_lim, double=double)
        req_arrsize = DeltaMatrix.estimate_array_size(lmin, lmax)
        limited = req_arrsize > max_arrsize
        if limited and high:
            lmin, _, _ = DeltaMatrix._get_array_params(lmax=lmax, arrsize=max_arrsize)
        elif limited:
            _, lmax, _ = DeltaMatrix._get_array_params(lmin=lmin, arrsize=max_arrsize)

        return lmin, lmax

    @classmethod
    def _set_wigner(cls, lmin, lmax, high=True, dtype=None):
        """
        Cache a DeltaMatrix, respecting memory limits.

        If a DeltaMatrix is already cached, it will use that as a starting point.

        Will recompute everything if the data type changes.
        """
        if lmax > cls.maximum_el:
            raise ValueError("Cannot construct DeltaMatrix within given memory limits.")
        dtype_set = (dtype is not None)
        lmin, lmax = cls._limit_lmin_lmax(lmin, lmax, high=high, dtype=dtype)
        if (cls.current_dmat is None or (dtype_set and cls.current_dmat.dtype != dtype)):
            cls.current_dmat = DeltaMatrix(lmax, lmin=lmin, dtype=dtype)
        elif (cls.current_dmat.lmax < lmax) or (cls.current_dmat.lmin > lmin):
            cls.current_dmat = DeltaMatrix(lmax, lmin=lmin, dmat=cls.current_dmat, dtype=dtype)

    @classmethod
    def wigner_d(cls, el, m1, m2, theta, lmax=None, lmin=None, double_prec=None):
        theta = np.atleast_1d(theta)
        if lmin is None:
            lmin = 0
        if lmax is None:
            lmax = el

        if el < lmin:
            raise ValueError(f"el {el} is less than lmin {lmin}")

        dtype = None
        if double_prec is not None:
            if double_prec:
                dtype = np.float64
            else:
                dtype = np.float32
        cls._set_wigner(lmin, lmax, dtype=dtype)

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
    def spin_spherical_harmonic(cls, spin, el, em, theta, phi,
                                lmin=None, lmax=None, double_prec=None):
        theta = np.asarray(theta)
        phi = np.asarray(phi)

        if not theta.shape == phi.shape:
            raise ValueError("theta and phi must have the same shape.")

        return (-1.)**(em) * np.sqrt((2 * el + 1) / (4 * np.pi))\
            * np.exp(1j * em * phi) * cls.wigner_d(el, em, -1 * spin,
                                                   theta, lmin=lmin, lmax=lmax,
                                                   double_prec=double_prec)

    @classmethod
    def clear_dmat(cls):
        """Delete cached DeltaMatrix."""
        cls.current_dmat = None


def wigner_d(el, m1, m2, theta, lmin=None, lmax=None, double_prec=None):
    """
    Evaluate the Wigner-d function (little-d) using cached values at pi/2.

    d^l_{m1, m2}(theta)

    Parameters
    ----------
    el, m1, m2: ints
        Indices of the d-function.
    theta: float or ndarray of float
        Angle argument(s) in radians.
    lmin: int
        Precompute the cached DeltaMatrix from some minimum el.
        (Optional. See notes.)
    lmax: int
        Precompute the cached DeltaMatrix up to some maximum el.
        (Optional. See notes.)
    double_prec: bool or None
        Compute the DeltaMatrix using doubles instead of singles.
        If set, this will enforce that the cached DeltaMatrix is in
        either single precision (False) or double (True).
        Otherwise, the cached DeltaMatrix will be kept at whatever
        precision it had previously, or will be made in single precision.
        Default None

    Returns
    -------
    complex or ndarray of complex
        Value of Wigner-d function.
        If multiple theta values are given, multiple values are returned.

    Notes
    -----
    Uses eqn 8 of McEwan and Wiaux (2011), which cites:
        A. F. Nikiforov and V. B. Uvarov (1991)

    When lmin/lmax are not set, the default behavior is to cache a DeltaMatrix
    covering 0 to about el in el.

    If larger el are required, the cached matrix is expanded.

    The total matrix size is limited so the cache doesn't exceed 200 MiB.
    This limit can be changed.
    """
    return HarmonicFunction.wigner_d(
        el, m1, m2, theta, lmin=lmin, lmax=lmax, double_prec=double_prec
    )


def spin_spherical_harmonic(spin, el, em, theta, phi, lmin=None, lmax=None, double_prec=None):
    """
    Evaluate the spin-weighted spherical harmonic.

    Obeys the standard numpy broadcasting for theta and phi.

    Parameters
    ----------
    spin: int
        Spin of the function.
    el, em: ints
        Spherical harmonic mode.
    theta: ndarray or float
        Colatitude(s) to evaluate to, in radians.
    phi: ndarray or float
        Azimuths to evaluate to, in radians.
    lmin: int
        Precompute the cached DeltaMatrix from some minimum el.
        (Optional. See notes on wigner_d.)
    lmax: int
        Precompute the cached Delta matrix up to some maximum el.
        (Optional. See notes on wigner_d.)
    double_prec: bool or None
        Compute the DeltaMatrix using doubles instead of singles.
        (See docstring for wigner_d for details).
        Default None

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
    return HarmonicFunction.spin_spherical_harmonic(
        spin, el, em, theta, phi, lmin=lmin, lmax=lmax, double_prec=double_prec
    )


def get_cached_dmat():
    """Return currently cached DeltaMatrix."""
    return HarmonicFunction.current_dmat


def clear_cached_dmat():
    """Delete cached DeltaMatrix."""
    HarmonicFunction.clear_dmat()


def set_cache_mem_limit(maxmem):
    """
    Set the memory limit for cached DeltaMatrix values.

    Defaults to 200 MiB.

    Also estimates the maximum el that can be contained in this limit by itself.

    Parameters
    ----------
    maxmem: float
        Memory usage limit in MiB for cached DeltaMatrix array.
        Defaults to 200, corresponding with about lmax ~ 540 for lmin = 0.
    """
    HarmonicFunction.set_cache_mem_limit(maxmem)


def get_cache_details():
    """
    Details of the cached DeltaMatrix.

    Returns
    -------
    dict:
        A dictionary containing:
            * cache_mem_limit
                Limit in MiB of the cached DeltaMatrix
            * maximum_el
                Maximum el mode that can be supported by itself
                in the cached DeltaMatrix, under the current limit.

                This is not the same as lmax. Each el mode "block"
                in the matrix consists of (el + 1) * (el + 2) / 2 floats.
                This is the maximum el whose block can fit in the memory limit.
        The following will be included only if a DeltaMatrix is cached:
            * lmin
                Minimum el mode in the cached DeltaMatrix
            * lmax
                Maximum el mode in the cached DeltaMatrix
            * size
                Number of floats in the cached DeltaMatrix
    """
    return HarmonicFunction.get_cache_details()


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

    Also at risk of an integer overflow for el > 10.
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

    if np.isscalar(res) and (np.isclose((theta / 2) % np.pi, 0)):
        res = 0.0
    else:
        res[np.isclose((theta / 2) % np.pi, 0)] = 0.0    # Singularities

    if res.size == 1:
        return complex(res)

    return res
