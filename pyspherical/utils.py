"""Utility functions."""

import numpy as np

from numba import jit, int32, types

__all__ = ['resize_axis', 'tri_ravel', 'unravel_lm', 'ravel_lm', 'get_grid_sampling']


def resize_axis(arr, size, mode='zero', axis=0):
    """
    Resize an axis of the array, either truncating or zero-padding.

    The "mode" keyword determines how this is done.

    Parameters
    ----------
    arr: ndarray
        Array to resize
    size: int
        New size for the axis.
    mode: str
        If the new array is zero-padded, where to put the old data:
            * 'zero' : Put zeros in the middle of the axis (center data on zero)
            * 'start' : Put zeros at the end of the axis.
            * 'center': Evenly fill data on both sides.
        Defaults to "zero".
    axis: int
        Which axis to resize

    Returns
    -------
    arr: ndarray
        Input array with the specified axis padded or truncated.

    Notes
    -----
    The 'zero' mode can be used to zero-pad an FFT-transformed array before
    applying an inverse transform.
    """
    shape = list(arr.shape)
    shape[axis] = size

    new = np.zeros(tuple(shape), dtype=arr.dtype)

    oldodd = arr.shape[axis] % 2 == 1
    newodd = size % 2 == 1

    L = arr.shape[axis]
    if oldodd:
        center = (L - 1) // 2   # This stays on the left.
    else:
        center = L // 2
    if newodd:
        newcent = (size - 1) // 2
    else:
        newcent = size // 2

    _arr = np.swapaxes(arr, axis, 0)
    _new = np.swapaxes(new, axis, 0)

    if mode == 'zero':
        limit = np.min([center, newcent])
        if oldodd:
            _new[:limit + 1, ...] = _arr[:limit + 1, ...]
            _new[-1:-limit - 1:-1, ...] = _arr[-1:-limit - 1:-1, ...]
        else:
            _new[:limit, ...] = _arr[:limit, ...]
            _new[-1:-limit - 1:-1, ...] = _arr[-1:-limit - 1:-1, ...]

    elif mode == 'start':
        limit = np.min([size, L])
        _new[:limit, ...] = _arr[:limit, ...]
    elif mode == 'center':
        if size <= L:
            # Truncating
            base = int(np.floor((L - size) / 2))
            _new[:, ...] = _arr[base:base + size, ...]
        else:
            base = int(np.floor((size - L) / 2))
            _new[base:base + L, ...] = _arr[:, ...]

    new = np.swapaxes(_new, 0, axis)

    return new


# Index raveling/unraveling

@jit(int32(int32, int32, int32), nopython=True)
def tri_ravel(l, m1, m2):
    """Ravel indices for the 'stack of triangles' ordering."""
    # m1 must be >= m2
    if m1 < m2 or m1 > l or m2 > l or m1 < 0 or m2 < 0:
        raise ValueError("Invalid indices")
    base = l * (l + 1) * (l + 2) // 6
    offset = (l - m1) * (l + 3 + m1) // 2 + m2
    ind = base + offset
    return int(ind)


@jit(int32(int32), nopython=True)
def tri_base(l):
    """Minimum index for a given el block."""
    return tri_ravel(l, l, 0)


@jit(int32(int32), nopython=True)
def el_block_size(l):
    """Size needed for a given el in the dmat array."""
    return (l + 1) * (l + 2) // 2


@jit(int32(int32, int32), nopython=True)
def unravel_lm(el, m):
    """Get index from (el, em)."""
    return el * (el + 1) + m


@jit(types.Tuple((int32, int32))(int32), nopython=True)
def ravel_lm(ind):
    """Get (el, em) from index."""
    el = int(np.floor(np.sqrt(ind)))
    m = ind - el * (el + 1)
    return el, m


def get_grid_sampling(lmax=None, Nt=None, Nf=None):
    """
    Get sample positions for "grid", compatible with the MW transform methods.

    Parameters
    ----------
    lmax: int
        Maximum multipole moment (Optional if Nt/Nf are set.)
    Nt: int
        Number of samples in theta. (Optional, defaults to lmax)
    Nf: int
        Number of samples in phi for all colatitudes
        (Optional, defaults to (2 * lmax - 1))

    Returns
    -------
    thetas: ndarray
        Colatitudes in radians.
    phis: ndarray
        Azimuths in radians.
    """
    if (lmax is None) and (Nt is None or Nf is None):
        raise ValueError("Need to provide lmax if Nt and Nf are unset.")

    if Nt is None:
        Nt = lmax

    if Nf is None:
        if lmax is not None:
            Nf = 2 * lmax - 1

    dth = np.pi / (2 * Nt - 1)
    thetas = np.linspace(dth, np.pi, Nt, endpoint=True)
    phis = np.linspace(0, 2 * np.pi, Nf, endpoint=False)

    return thetas, phis
