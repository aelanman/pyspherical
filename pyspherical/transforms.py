
import numpy as np

from .wigner import HarmonicFunction
from .utils import resize_axis, unravel_lm

# -----------
# Forward transform
# -----------


def _theta_fft(Gm_th, thetas, lmax, lmin=0, spin=0):
    """
    Do transform over theta, using the approach in McEwen and Wiaux (2011).
    """

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
        # is in the middle of the array.
        return np.convolve(a, weights, mode='valid')

    Gmm = np.apply_along_axis(do_conv, 1, padFmm) * 2 * np.pi

    # Unshift the m' axis
    Gmm = np.fft.ifftshift(Gmm, axes=1)

    return Gmm


def _dmm_to_flm(dmm, lmax, spin):
    # TODO -- numba accelerate this.

    flm = np.zeros(lmax**2, dtype=complex)
    HarmonicFunction._set_wigner(lmax + 1)
    wig_d = HarmonicFunction.current_dmat

    for el in range(spin, lmax):
        prefac = np.sqrt((2 * el + 1) / (4 * np.pi))
        for m in range(-el, el + 1):
            ind = unravel_lm(el, m)
            prefac2 = (-1)**(el + m) * (1j)**(m + spin)
            for mp in range(-el, el + 1):
                flm[ind] += prefac * prefac2 * wig_d[el, mp, m] * \
                    wig_d[el, spin, mp] * dmm[m, mp]

                # NOTE -- going -spin -> spin fixed (up to a factor of (-1) the nonzero spin test. Investigate

    return flm


def _do_fwd_transform_on_grid(dat, phis, thetas, lmax, lmin=0, ph_ax=0, th_ax=1, spin=0):
    """
    Do forward transform, assuming a regular grid in both theta and phi.
    """

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
            "Non-equispaced latitudes are not yet supported.")
    flm = _dmm_to_flm(dmm, lmax, spin)

    return flm


def _do_fwd_transform_nongrid(dat, phis, thetas, lmax, lmin, spin):
    """
    Forward transform. Assumes isolatitude samples with equal
    spacing in phi on each ring.
    """
    # TODO -- write a separate function for healpix, taking advantage
    #    of its regularity for grouping pixels into rings.

    dat = dat.flatten()
    phis = phis.flatten()
    thetas = thetas.flatten()

    un_thetas, lat_ind = np.unique(thetas, return_inverse=True)
    Nlats = un_thetas.size

    # phi to m, per ring
    dm_th = np.zeros((2 * lmax - 1, Nlats), dtype=complex)

    for th_i, th in enumerate(un_thetas):
        ring = lat_ind == th_i
        dat_i = dat[ring]

        Nf = dat_i.size
        dm_th[:, th_i] = resize_axis(np.fft.fft(
            dat_i) / Nf, (2 * lmax - 1), mode='zero')

    # theta to m'
    dth = np.diff(un_thetas)
    if np.allclose(dth, dth[0]):
        dmm = _theta_fft(dm_th, un_thetas, lmax, lmin, spin)
    else:
        raise NotImplementedError(
            "Non-equispaced latitudes are not yet supported.")

    flm = _dmm_to_flm(dmm, lmax, spin)

    return flm


def forward_transform(dat, phis, thetas, lmax, lmin=0, spin=0):
    """

    Transform sampled data to spin-weighted spherical harmonics.

    """
    # TODO -- Credit the FFT-based transforms to:
    #   Eqns 9 - 12
    #  http://scripts.iucr.org/cgi-bin/paper?S0108767306017478
    #   Ref 41 of "A novel sampling theorem on the sphere"

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
                         "\n\tthetas.shape = {}".format(str(dat.shape), str(phis.shape), str(thetas.shape)))

    if grid:
        return _do_fwd_transform_on_grid(dat, phis, thetas, lmax, lmin, fax, tax, spin)

    else:
        return _do_fwd_transform_nongrid(dat, phis, thetas, lmax, lmin, spin)


# -----------
# Inverse transform
# -----------

def _flm_to_fmm(flm, lmax, spin):
    # Transform components flm to Fmm (Fourier-transformed data).

    fmm = np.zeros(((2 * lmax - 1), (2 * lmax - 1)), dtype=complex)
    HarmonicFunction._set_wigner(lmax + 1)
    wig_d = HarmonicFunction.current_dmat

    for li, el in enumerate(range(spin, lmax)):
        prefac = (-1)**spin * np.sqrt((2 * el + 1) / (4 * np.pi))
        for m in range(-el, el + 1):
            prefac2 = (1j)**(-m - spin) * flm[unravel_lm(el, m)]
            for mp in range(-el, el + 1):
                fmm[m, mp] += prefac * prefac2 * \
                    wig_d[el, m, mp] * wig_d[el, -spin, mp]
    return fmm


def _theta_ifft(fmm, lmax, spin, Nt=None, offset=None):
    # Transform Fmm' to spherical coordinates via inverse FFTs in theta and phi.
    if Nt is None:
        Nt = lmax

    if offset is None:
        offset = np.pi / (2 * Nt - 1)

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
    """
    Do inverse transform assuming regular grid in theta and phi.
    """

    Fmm = _flm_to_fmm(flm, lmax, spin)

    # Transform over theta
    theta_is_equi = np.allclose(
        thetas[1] - thetas[0], thetas[2:] - thetas[1:-1])

    if theta_is_equi:
        Fm_th = _theta_ifft(
            Fmm, lmax, spin, offset=thetas[0], Nt=thetas.size)
    else:
        raise NotImplementedError(
            "Non-equispaced latitudes are not yet supported.")

    # Transform over phi
    Nf = phis.size
    Fm_th = resize_axis(Fm_th, Nf, axis=0)
    dat = np.fft.ifft(Fm_th, axis=0) * Nf

    return dat


def _do_inv_transform_nongrid(flm, phis, thetas, lmax, lmin=0, spin=0):
    """
    Inverse transform. Assumes isolatitude samples with equal
    spacing in phi on each ring.
    """

    phis = phis.flatten()
    thetas = thetas.flatten()

    Fmm = _flm_to_fmm(flm, lmax, spin)

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
            "Non-equispaced latitudes are not yet supported.")

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


def inverse_transform(flm, phis=None, thetas=None, lmax=None, lmin=0, spin=0, Nt=None, Nf=None, grid=None):
    """
    Evaluate spin-weighted spherical harmonic components to a sphere.

    phis/thetas overrides Nt/Nf

    """
    # TODO -- Credit the FFT-based transforms to:
    #   Eqns 9 - 12
    #  http://scripts.iucr.org/cgi-bin/paper?S0108767306017478
    #   Ref 41 of "A novel sampling theorem on the sphere"

    if lmax is None:
        lmax = int(np.sqrt(flm.size))

    if np.sqrt(flm.size) % 1 != 0:
        raise ValueError("Invalid flm shape: {}".format(str(flm.shape)))

    if thetas is None:
        if Nt is None:
            Nt = lmax
        dth = np.pi / (2 * Nt - 1)
        thetas = np.linspace(dth, np.pi, Nt, endpoint=True)

    if phis is None:
        if Nf is None:
            Nf = 2 * lmax - 1
        phis = np.linspace(0, 2 * np.pi, Nf, endpoint=False)

    phis = np.asarray(phis)
    thetas = np.asarray(thetas)

    Nt = thetas.size
    Nf = phis.size

    # Guess from coordinate shapes/sizes if they represent axes of a grid or
    # fully cover the desired sampling.
    if grid is None:
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
