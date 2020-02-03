
import numpy as np
from wigner import dMat

from utils import resize_axis


# Structure:
#   Functions for separate transform methods/parts of transforms.
#   Wrapper that lets you select which method to use.
#   Function to provide appropriate sampling if transforming a function.


def unravel_lm(el, m):
    return el*(el + 1) + m


def ravel_lm(ind):
    el = int(np.floor(np.sqrt(ind)))
    m = ind - el * (el + 1)

    return el, m


def _theta_fft(Gm_th, thetas, lmax, lmin=0, s=0):
    """
    Do transform over theta, using the approach in McEwen and Wiaux (2011).
    """

    Nt = thetas.size
    dth = thetas[1] - thetas[0]

    # Apply periodic extension in theta.
    Gm_th = np.pad(Gm_th, [(0,0), (0, Nt-1)],  mode='constant')
    em = np.fft.ifftshift(np.arange(-(lmax-1), lmax))
    Gm_th[:, Nt:] = ((-1.0)**(s + em) * Gm_th[:, 2*Nt-2 - np.arange(Nt, 2*Nt - 1)].T).T
    
    Fmm = np.fft.fft(Gm_th, axis=1) / (2* Nt - 1)

    if thetas[0] > 0:
#       Fmm = (Fmm.T *  np.exp(-1j * em * dth)).T
        Fmm = (Fmm.T * np.exp(-1j * em * np.pi/(2*Nt -1))).T
        # TODO Careful with this. If Fmm's axes are the same size the behavior is inconsistent.

    ## Truncate/zero-pad the m' axis, then convolve with weights over m'
    padFmm = resize_axis(Fmm, (2*lmax - 1), axis=1, mode='zero')
    padFmm = np.fft.fftshift(padFmm, axes=1)
    
    ## Get the weights:
    def weight(em):
        if np.abs(em) == 1:
            return np.sign(em) * np.pi * 1j / 2.
        if em % 2 == 1:
            return 0
        return 2/(1 - em**2)
    
    em_ext = range(-2*(lmax - 1), 2 * (lmax - 1)+1)
    ite = (weight(mm) for mm in em_ext)
    weights = np.fromiter(ite, dtype=complex)[::-1]
    
    def do_conv(a):
        # The "valid" option only works with these Fourier-transformed
        # quantities if they have been FFT-shifted, such that the 0 mode
        # is in the middle of the array.
        return np.convolve(a, weights, mode='valid')
    
    Gmm = np.apply_along_axis(do_conv, 1, padFmm) * 2*np.pi
    
    # Unshift the m' axis
    Gmm = np.fft.ifftshift(Gmm, axes=1)

    return Gmm

def _dmm_to_flm(dmm, lmax, spin):
    ## TODO -- numba accelerate this.

    flm = np.zeros((1+lmax)**2, dtype=complex)
    wig_d = dMat(lmax)

    for el in range(lmax + 1):
        prefac = np.sqrt((2*el + 1)/(4*np.pi))
        for m in range(-el, el+1):
            ind = unravel_lm(el, m)
            prefac2 = (-1)**(el+m) * (1j)**(m+s)
            for mp in range(-el, el+1):
                flm[ind] += prefac * prefac2 * wig_d[el, mp, m] * wig_d[el, -s, mp] * dmm[m, mp]

    return flm
        

def _do_transform_on_grid(dat, phis, thetas, lmax, lmin=0, ph_ax=0, th_ax=1, spin=0):
    """
    Do forward transform, assuming a regular grid in both theta and phi.
    """

    if th_ax == 0:
        # Underlying functions expect the theta and phi axes to be 1 and 0, resp.
        dat = dat.T

    Nf, Nt = dat.shape

    # Transform phi to m and pad/truncate.
    dm_th = np.fft.fft(dat, axis=0) / Nf
    dm_th = resize_axis(dm_th, (2*lmax - 1), mode='zero', axis=0)
    # TODO Is a phase correction necessary for the phi transform?

    # Transform theta to m'.
    # If evenly-spaced in theta, can use an FFT.

    dth = np.diff(thetas)
    if np.allclose(dth, dth[0]):
        dmm = _theta_fft(dm_th, thetas, lmax, lmin, spin)
    else:
        raise NotImplementedError("Non-equispaced latitudes are not yet supported.")
    
    flm = _dmm_to_flm(dmm, lmax, spin) 

    return flm 


def _do_transform_nongrid(dat, phis, thetas, lmax, lmin, spin):
    """
    Forward transform. Assumes isolatitude samples with equal
    spacing in phi on each ring.
    """
    # TODO -- write a separate function for healpix, taking advantage
    #    of its regularity for grouping pixels into rings.

    dat = dat.flatten()
    phis = phis.flatten()
    thetas = thetas.flatten()

    un_thet, lat_ind = np.unique(thetas, return_inverse=True)
    Nlats = un_thet.size

    # phi to m, per ring
    dm_th = np.zeros((2*lmax - 1, Nlats), dtype=complex)

    for th_i, th in enumerate(un_thet):
        ring = lat_ind == th_i
        phi_i = phis[ring]
        dat_i = dat[ring]

        # TODO Is a phase correction necessary for the phi transform?
        Nf = dat_i.size
        dm_th[:, th_i] = resize_axis(np.fft.fft(dat_i)/Nf, (2*lmax - 1), mode='zero')


    # theta to m'
    dth = np.diff(un_thet)
    if np.allclose(dth, dth[0]):
        dmm = _theta_fft(dm_th, un_thet, lmax, lmin, spin)
    else:
        raise NotImplementedError("Non-equispaced latitudes are not yet supported.")

    flm = _dmm_to_flm(dmm, lmax, spin)

    return flm


def forward_transform(dat, phis, thetas, lmax, lmin=0, spin=0):
    """
  
    """

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
        return _do_transform_on_grid(dat, phis, thetas, lmax, lmin, fax, tax, spin)

    else:
        return _do_transform_nongrid(dat, phis, thetas, lmax, lmin, spin)

 


if __name__ == '__main__':
    import pylab as pl
    from scipy.special import sph_harm

    s = 0   # Spin
    lmax = 10   # Maximum el
    Nt = 701   # Number of samples in theta (must be odd)
    Nf = 401  # Samples in phi
#    Nf = 2*lmax-1
#    Nt = lmax
    
    # Define samples
    dth = np.pi/(2*Nt-1)
    theta = np.linspace(dth, np.pi, Nt, endpoint=True)
    phi = np.linspace(0, 2*np.pi, Nf, endpoint=False)
    
    # Data, shape (Nf , Nt)
    gtheta, gphi = np.meshgrid(theta, phi)
    dat = 10 * sph_harm(3,5, gphi, gtheta)
    dat += 11 * sph_harm(4,8, gphi, gtheta)

    #res = phi_fft(dat, phi, theta)
    flm = forward_transform(dat, gphi, gtheta, lmax)
    import IPython; IPython.embed()
