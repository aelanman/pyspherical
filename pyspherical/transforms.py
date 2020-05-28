"""
Fast methods for forward and inverse spin-weighted spherical harmonic transforms.

An extension of the methods presented in McEwen and Wiaux (2011) [MW], the transforms here
use FFTs and numba-acceleration to rapidly compute the spin-weighted spherical harmonic
decomposition of a discretely-sampled function on a sphere, or invert the decomposition back to
sampled values. The only requirement is that the sphere be sampled with evenly-spaced samples
in azimuth (phi) at each colatitude (theta).

Non-uniform sampling in theta will be supported in the future.
"""

import numpy as np
from numba import jit

from .wigner import HarmonicFunction, _access_element
from .utils import resize_axis, unravel_lm, get_grid_sampling


__all__ = ['forward_transform', 'inverse_transform']

# -----------------
# Forward transform
# -----------------


def _theta_fft(Gm_th, thetas, lmax, lmin=0, spin=0):

    Nt = thetas.size

    # Apply periodic extension in theta.
    Gm_th = np.pad(Gm_th, [(0, 0), (0, Nt - 1)], mode='constant')
    em = np.fft.ifftshift(np.arange(-(lmax - 1), lmax))
    Gm_th[:, Nt:] = ((-1.0)**(spin + em)
                     * Gm_th[:, 2 * Nt - 2 - np.arange(Nt, 2 * Nt - 1)].T).T

    Fmm = np.fft.fft(Gm_th, axis=1) / (2 * Nt - 1)

    # Truncate/zero-pad the m' axis
    padFmm = resize_axis(Fmm, (2 * lmax - 1), axis=1, mode='zero')

    # Apply phase offset, for thetas with nonzero origin.
    if thetas[0] > 0:
        padFmm = (padFmm * np.exp(-1j * em * np.pi / (2 * Nt - 1)))

    # Convolve with weights over m'

    # Need to shift to zero-centered Fourier ordering for the convolutions.
    padFmm = np.fft.fftshift(padFmm, axes=1)

    def weight(em):
        if np.abs(em) == 1:
            return np.sign(em) * np.pi * 1j / 2.
        if em % 2 == 1:
            return 0
        return 2 / (1 - em**2)

    em_ext = range(-2 * (lmax - 1), 2 * (lmax - 1) + 1)
    ite = (weight(mm) for mm in em_ext)
    weights = np.fromiter(ite, dtype=complex)[::-1]

    def do_conv(a):
        # The "valid" option only works with these Fourier-transformed
        # quantities if they have been FFT-shifted, such that the 0 mode
        # is in the center of the array.
        return np.convolve(a, weights, mode='valid')

    Gmm = np.apply_along_axis(do_conv, 1, padFmm) * 2 * np.pi

    # Unshift the m' axis
    Gmm = np.fft.ifftshift(Gmm, axes=1)

    return Gmm


@jit(nopython=True)
def _jit_dmm2flm(dmm, cur_lmax, spin, dmatarr, lmin, flm, out_lmin):
    # Faster evaluation.
    # flm = output array, written to directly.

    loopmin = max(out_lmin, spin, lmin)
    for el in range(loopmin, cur_lmax):
        prefac = np.sqrt((2 * el + 1) / (4 * np.pi))
        for m in range(-el, el + 1):
            ind = unravel_lm(el, m) - out_lmin**2
            # The MW paper was missing a factor of (-1)**(m+spin) here.
            prefac2 = (-1)**(m + 2 * spin) * (1j)**(m + spin)
            # mp == 0
            flm[ind] += _access_element(el, 0, m, dmatarr, lmin=lmin) \
                * _access_element(el, 0, -spin, dmatarr, lmin=lmin) * dmm[m, 0]
            negfac = (-1)**(m + spin)
            for mp in range(-el, 0):
                dels = _access_element(el, mp, m, dmatarr, lmin=lmin) \
                    * _access_element(el, mp, -spin, dmatarr, lmin=lmin)
                flm[ind] += dels * (dmm[m, mp] + negfac * dmm[m, -mp])
            flm[ind] *= prefac * prefac2


def _dmm_to_flm(dmm, lmax, spin, flm=None, lmin=0, out_lmin=0):
    # Recursion is to handle potential memory-limited situations:
    #  - If the cached Wigner matrix has an lmax < less than given lmax,
    #    then rerun with the smaller lmax and then with the remainder.
    #  - Otherwise, fill flm using the JIT-compiled function above.
    if flm is None:
        out_lmin = lmin
        flm = np.zeros(lmax**2 - out_lmin**2, dtype=complex)
    HarmonicFunction._set_wigner(lmin, lmax, high=False)
    wig_d = HarmonicFunction.current_dmat
    if wig_d.lmax < lmax:
        _dmm_to_flm(dmm, wig_d.lmax, spin, flm=flm, lmin=lmin)
        _dmm_to_flm(dmm, lmax, spin, flm=flm, lmin=wig_d.lmax)
    else:
        _jit_dmm2flm(dmm, lmax, spin, wig_d._arr, wig_d.lmin, flm, out_lmin)

    return flm


def _do_fwd_transform_on_grid(dat, phis, thetas, lmax, lmin=0, ph_ax=0, th_ax=1, spin=0):
    """Do forward transform, assuming a regular grid in both theta and phi."""
    if th_ax == 0:
        # Underlying functions expect the theta and phi axes to be 1 and 0, resp.
        dat = dat.T

    Nf, Nt = dat.shape

    # Transform phi to m and pad/truncate.
    dm_th = np.fft.fft(dat, axis=0) / Nf
    dm_th = resize_axis(dm_th, (2 * lmax - 1), mode='zero', axis=0)

    # Transform theta to m'.
    # If evenly-spaced in theta, can use an FFT.

    dth = np.diff(thetas)
    if np.allclose(dth, dth[0]):
        dmm = _theta_fft(dm_th, thetas, lmax, lmin, spin)
    else:
        raise NotImplementedError(
            "Non-uniform latitude spacing is not yet supported.")
    flm = _dmm_to_flm(dmm, lmax, spin, lmin=lmin)

    return flm


def _do_fwd_transform_nongrid(dat, phis, thetas, lmax, lmin, spin):
    """
    Do forward transform without assuming the same phi sampling for all colatitudes.

    Assumes uniformly-spaced samples in phi on each colatitude ring.
    """
    dat = dat.flatten()
    phis = phis.flatten()
    thetas = thetas.flatten()

    un_thetas, lat_ind = np.unique(thetas, return_inverse=True)
    Nlats = un_thetas.size

    # phi to m, per ring
    dm_th = np.zeros((2 * lmax - 1, Nlats), dtype=complex)

    em = np.fft.ifftshift(np.arange(-(lmax - 1), lmax))
    for th_i, th in enumerate(un_thetas):
        ring = lat_ind == th_i
        dat_i = dat[ring]
        Nf = dat_i.size
        dm_th[:, th_i] = resize_axis(np.fft.fft(
            dat_i) / Nf, (2 * lmax - 1), mode='zero')
        # Apply phase offset when the ring doesn't start at phi = 0
        phase = np.exp(-1j * em * np.min(np.abs(phis[ring])))
        dm_th[:, th_i] *= phase

    # theta to m'
    dth = np.diff(un_thetas)
    if np.allclose(dth, dth[0]):
        dmm = _theta_fft(dm_th, un_thetas, lmax, lmin, spin)
    else:
        raise NotImplementedError(
            "Non-uniform latitude spacing is not yet supported.")

    flm = _dmm_to_flm(dmm, lmax, spin, lmin=lmin)

    return flm


def forward_transform(dat, phis, thetas, lmax, lmin=0, spin=0):
    """
    Transform sampled data to spin-weighted spherical harmonics.

    Parameters
    ----------
    dat: ndarray of float or complex
        Sampled data array.
        Shape can be:
            * dat.shape = (phis.size, thetas.size)
            * dat.shape = (thetas.size, phis.size)
            * dat.shape = phis.shape = theta.shape
        In the first two cases, it is assumed that the same
        sampling in phi is done at all co-latitudes.
    phis: ndarray of float
        Sample azimuths in radians.
    thetas: ndarray of float
        Sample co-latitudes in radians.
    lmax: int
        Maximum multipole mode.
    lmin: int
        Minimum multipole mode to compute. (Optional, default 0)
    spin: int
        Spin of the transform.

    Returns
    -------
    flm: ndarray of complex
        Spherical-harmonic components, shape ((lmax-1)**2)
        Ordered as:
            el  = 0  1  1  1  2  2  2  2  2
            em  = 0 -1  0  1 -2 -1  0  1  2
            ind = 0  1  2  3  4  5  6  7  8
        Modes with el < spin will be set to zero.
    """

    phis = np.asarray(phis)
    thetas = np.asarray(thetas)
    grid = True
    if dat.shape == (phis.size, thetas.size):
        fax, tax = 0, 1
    elif dat.shape == (thetas.size, phis.size):
        fax, tax = 1, 0
    elif dat.shape == thetas.shape == phis.shape:
        grid = False
    else:
        raise ValueError("Data shapes inconsistent:"
                         "\n\tdat.shape = {}"
                         "\n\tphis.shape = {}"
                         "\n\tthetas.shape = {}".format(
                             str(dat.shape), str(phis.shape), str(thetas.shape))
                         )

    if grid:
        return _do_fwd_transform_on_grid(dat, phis, thetas, lmax, lmin, fax, tax, spin)

    else:
        return _do_fwd_transform_nongrid(dat, phis, thetas, lmax, lmin, spin)


# -----------------
# Inverse transform
# -----------------

@jit(nopython=True)
def _jit_flm2fmm(flm, cur_lmax, spin, dmatarr, lmin, fmm, in_lmin):
    loopmin = max(lmin, in_lmin, spin)
    for el in range(loopmin, cur_lmax):
        prefac = (-1)**spin * np.sqrt((2 * el + 1) / (4 * np.pi))
        for m in range(-el, el + 1):
            prefac2 = (1j)**(-m - spin) * flm[unravel_lm(el, m) - in_lmin**2]
            # mp == 0
            fmm[m, 0] += prefac * prefac2 * _access_element(el, 0, m, dmatarr, lmin=lmin) \
                * _access_element(el, 0, -spin, dmatarr, lmin=lmin)
            negfac = (-1)**(m + spin)
            for mp in range(1, el + 1):
                val = prefac * prefac2 * _access_element(el, m, mp, dmatarr, lmin=lmin) \
                    * _access_element(el, -spin, mp, dmatarr, lmin=lmin)
                fmm[m, mp] += val
                fmm[m, -mp] += val * negfac


def _flm_to_fmm(flm, lmax, spin, fmm=None, lmin=0, in_lmin=0):
    if fmm is None:
        in_lmin = lmin
        fmm = np.zeros(((2 * lmax - 1), (2 * lmax - 1)), dtype=complex)
    HarmonicFunction._set_wigner(lmin, lmax, high=False)
    wig_d = HarmonicFunction.current_dmat
    if wig_d.lmax < lmax:
        _flm_to_fmm(flm, wig_d.lmax, spin, fmm=fmm, lmin=lmin, in_lmin=in_lmin)
        _flm_to_fmm(flm, lmax, spin, fmm=fmm, lmin=wig_d.lmax, in_lmin=in_lmin)
    else:
        _jit_flm2fmm(flm, lmax, spin, wig_d._arr, wig_d.lmin, fmm, in_lmin)

    return fmm


def _theta_ifft(fmm, lmax, spin, Nt=None, offset=None):
    # offset = np.pi / (2 * Nt - 1) for MW sampling

    em = np.fft.ifftshift(np.arange(-(lmax - 1), lmax))

    # Apply phase offset
    fmm *= np.exp(1j * em * offset)

    # Apply ifft over theta
    fmm = resize_axis(fmm, 2 * Nt - 1, axis=1)
    Fm_th = np.fft.ifft(fmm, axis=1) * (2 * Nt - 1)

    # Cut the periodic extension in theta
    Fm_th = Fm_th[:, :Nt]

    return Fm_th


def _do_inv_transform_on_grid(flm, phis, thetas, lmax, lmin=0, spin=0):
    """Do inverse transform assuming regular grid in theta and phi."""
    Fmm = _flm_to_fmm(flm, lmax, spin, lmin=lmin)

    # Transform over theta
    theta_is_equi = np.allclose(
        thetas[1] - thetas[0], thetas[2:] - thetas[1:-1])

    if theta_is_equi:
        Fm_th = _theta_ifft(
            Fmm, lmax, spin, offset=thetas[0], Nt=thetas.size)
    else:
        raise NotImplementedError(
            "Non-uniform latitude spacing is not yet supported.")

    # Transform over phi
    Nf = phis.size
    Fm_th = resize_axis(Fm_th, Nf, axis=0)
    dat = np.fft.ifft(Fm_th, axis=0) * Nf

    return dat


def _do_inv_transform_nongrid(flm, phis, thetas, lmax, lmin=0, spin=0):
    """
    Do inverse transform without assuming the same phi sampling for all colatitudes.

    Assumes uniformly-spaced samples in phi on each colatitude ring.
    """
    phis = phis.flatten()
    thetas = thetas.flatten()

    Fmm = _flm_to_fmm(flm, lmax, spin, lmin=lmin)

    un_thetas, lat_ind = np.unique(thetas, return_inverse=True)
    Nlats = un_thetas.size

    # Transform over theta
    theta_is_equi = np.allclose(
        un_thetas[1] - un_thetas[0], un_thetas[2:] - un_thetas[1:-1])

    if theta_is_equi:
        Fm_th = _theta_ifft(
            Fmm, lmax, spin, offset=un_thetas[0], Nt=Nlats
        )
    else:
        raise NotImplementedError(
            "Non-uniform latitude spacing is not yet supported.")

    # Transform m to phi, per ring.
    #   Need to apply a phase offset if the 0th index is not at phi = 0
    em = np.fft.ifftshift(np.arange(-(lmax - 1), lmax))
    dat = np.zeros(phis.size, dtype=complex)
    for th_i, th in enumerate(un_thetas):
        ring = lat_ind == th_i
        phi_i = np.sort(phis[ring])
        Nf = phi_i.size
        phase = np.exp(1j * em * phi_i[0])
        ring_dat = phase * Fm_th[:, th_i]
        dat[ring] = np.fft.ifft(resize_axis(ring_dat, Nf)) * Nf

    return dat


def inverse_transform(flm, phis=None, thetas=None, lmax=None, lmin=0, spin=0, Nt=None, Nf=None):
    """
    Evaluate spin-weighted spherical harmonic components to a sphere.

    Without any optional parameters, this assumes the sampling described in McEwan and Wiaux (2011).

    Parameters
    ----------
    flm: ndarray of complex
        Spherical harmonic components in the ordering returned by forward_transform.
    phis: ndarray of float
        Sample azimuths in radians. Overrides Nf keyword. (Optional)
        Defaults to Nf uniform samples in [0,  2 pi).
    thetas: ndarray of float
        Sample co-latitudes in radians. Overrides Nt keyword. (Optional)
        Defaults to Nt uniform samples in [dth, pi], where dth = pi/(2 * Nt - 1)
    lmax: int
        Maximum multipole mode. (Optional)
        Defaults to sqrt(len(flm)).
    lmin: int
        Minimum multipole mode to compute. (Optional, default 0)
    spin: int
        Spin of the transform.
    Nt: int
        Number of samples in theta. (Optional, defaults to lmax)
    Nf: int
        Number of samples in phi for all colatitudes (Optional, defaults to (2 * lmax - 1))

    Returns
    -------
    dat: ndarray of complex
        Evaluation of the spherical harmonic components to the sample positions.


    Notes
    -----
    The sampling given through phi/theta can either be the axes of a grid
    (i.e., identical phi sampling for all thetas) or specify the individual phi/theta positions
    for each sample. In the first case, dat.shape = (phi.size, theta.size). In the latter case,
    dat.shape = phi.shape = theta.shape.

    This function cannot be used to evaluate the spherical harmonic components to arbitrary
    positions on the sphere. For that, the function
    :meth:`HarmonicFunction.spin_spherical_harmonic()` is provided.
    """
    if lmax is None:
        lmax = int(np.sqrt(flm.size))

    if np.sqrt(flm.size + lmin**2) % 1 != 0:
        raise ValueError("Invalid flm shape: {}".format(str(flm.shape)))

    _thetas, _phis = get_grid_sampling(lmax, Nt, Nf)
    if thetas is None:
        thetas = _thetas
    if phis is None:
        phis = _phis

    phis = np.asarray(phis)
    thetas = np.asarray(thetas)

    Nt = thetas.size
    Nf = phis.size

    # Guess from coordinate shapes/sizes if they represent axes of a grid or
    # fully cover the desired sampling.
    grid = True
    if not np.unique(thetas).size == thetas.size:
        grid = False
        if not phis.shape == thetas.shape:
            raise ValueError("It looks like the coordinates are non-grid, "
                             "but their shapes are inconsistent:"
                             "\n\tphis.shape = {}"
                             "\n\tthetas.shape = {}".format(str(phis.shape), str(thetas.shape)))

    if grid:
        return _do_inv_transform_on_grid(flm, phis, thetas, lmax, lmin, spin)

    else:
        return _do_inv_transform_nongrid(flm, phis, thetas, lmax, lmin, spin)
