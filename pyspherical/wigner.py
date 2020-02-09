
import numpy as np
from numba import jit, int32

from .utils import tri_ravel

@jit(nopython=True)
def _dmat_eval(lmin, lmax, arr):
    # Evaluate the values of the Wigner-d matrices at pi/2.
    # arr = linear array, modified in-place
    # TODO -- Modify to use lmin --- What "prior steps" need to be stored?
    arr[0] = 1.0
    for el in range(1, lmax+1):
        dll0 = -np.sqrt((2*el - 1)/float(2*el)) * arr[tri_ravel(el - 1, el - 1, 0)]
        arr[tri_ravel(el, el, 0)] = dll0
        for m2 in range(0, el+1):
            if m2 > 0:
                dllm = np.sqrt(
                        (el/2.) * (2* el - 1)/ ((el + m2)*(el + m2 - 1))
                        ) * arr[tri_ravel(el - 1, el - 1, m2 - 1)]
                arr[tri_ravel(el, el, m2)] = dllm
            for m1 in range(el-1, m2-1, -1):
                fac1 = (2 * m2)/np.sqrt( (el - m1) * (el + m1 + 1)) * arr[tri_ravel(el, m1 + 1, m2)]
                fac2 = 0.0
                if (m1 + 2) <= el:
                    fac2 = np.sqrt( ( (el - m1 - 1) * (el + m1 + 2) )/ ( (el - m1) * (el + m1 + 1) ) ) * arr[tri_ravel(el, m1 + 2, m2)]
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



class DeltaMatrix(object):
    """
    Wigner-d functions evaluated at pi/2.

    Only stores values for m1, m2 >= 0. Other values are returned by symmetry.

    Based on the methods in:
        S, Trapani, and Navaza J. Acta Crystallographica Section A, vol. 62, no. 4, 2006, pp. 262–69.


    """
    ## TODO
    #       *Have shown that numba acceleration is helping.
    #       *Test against another package -- mathematica / spherical_functions 
    #       Handling slices?
    #       *Numba acceleration on recursive calc and indexing.
    #       *Accessing non-stored values --- using symmetry relations
    #       *Cite the paper defining the recursive relations (https://arxiv.org/pdf/1110.6298.pdf)
    #       Check recursive stability for large l -- need to compare against a different calculation.
    #       Limited storage case -- only store after an lmin and up to lmax.
    #       "unravel" method
    #       !! Move the ravel/unravel methods to utils

    ##   Eqns 9 - 12
    ##  http://scripts.iucr.org/cgi-bin/paper?S0108767306017478
    ##   Ref 41 of "A novel sampling theorem on the sphere"

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
        """ Estimate size of the flattened array needed to store Delta_lmm values."""
        return (1 + lmax - lmin) * (6 + 5 * lmax + lmax**2 + 4
                * lmin + lmax * lmin + lmin**2) // 6

    @classmethod
    def ravel_index(cls, l, m1, m2): 
        # Ravel indices for the 'stack of triangles' format.
        return tri_ravel(l, m1, m2)

    def __getitem__(self, index):
        # Access stored elements, or use 
        # symmetry relations for non-stored elements.

        (l, m1, m2) = index
        return _access_element(l, m1, m2, self._arr)


current_dmat = None


@jit(nopython=True)
def _get_matrix_elements(el, m1, m2, arr, outarr):
    """
    Get an array of Delta[el, mp, m1] * Delta[el, mp, m2]

    Results are written into "outarr".

    """
    for mi, mp in enumerate(range(0, el+1)):
        outarr[mi] = _access_element(el, mp, m1, arr) * _access_element(el, mp, m2, arr) 


def wigner_d(el, m1, m2, theta, lmax=None):
    """
    Evaluate the Wigner-d function (little-d).

    Caches values at pi/2.
    """
    global current_dmat
    theta = np.atleast_1d(theta)
    if (current_dmat is None) and (lmax is not None):
        current_dmat = DeltaMatrix(lmax)
    if (current_dmat is None) or (current_dmat.lmax < el):
        current_dmat = DeltaMatrix(el)

    mp = np.arange(1, el+1)
    exp_fac = np.exp(1j * mp[None, :] * theta[..., None])
    dmats = np.empty(el+1, dtype=float)
    _get_matrix_elements(el, m1, m2, current_dmat._arr, dmats)

    val = (1j) ** (m2 - m1) * (
            np.sum((exp_fac + (-1)**(m1 + m2 - 2*el) / exp_fac) * dmats[1:], axis=-1)
            + dmats[0]
            )
    return val.squeeze()

def spin_spherical_harmonic(el, em, spin, theta, phi, lmax=None):
    """
    Evaluate the spin-weighted spherical harmonic.
    """

    prefactor = (-1)**(spin) * np.sqrt((2 * el + 1)/(4*np.pi))
    exp = np.exp(-1j * em * phi)
    return prefactor * exp * wigner_d(el, em, -1*spin, theta, lmax=lmax)
