
import numpy as np
from utils import resize_axis
from wigner import dMat
from scipy.special import sph_harm
import pylab as pl

s = 0   # Spin
lmax = 10   # Maximum el
Nt = 701   # Number of samples in theta (must be odd)
Nf = 701 # Samples in phi

# Define samples
dth = np.pi/Nt
theta = np.linspace(dth, np.pi, Nt, endpoint=True)
phi = np.linspace(0, 2*np.pi, Nf, endpoint=False)

# Data, shape (Nf , Nt)
gtheta, gphi = np.meshgrid(theta, phi)
dat = 10 * sph_harm(3,5, gphi, gtheta)
dat += 10 * sph_harm(1,8, gphi, gtheta)


# First transform --- FFT phi to m, then zero pad or truncate to (2*lmax - 1)
Gm_th = np.fft.fft(dat, axis=0) * 1 / Nf
Gm_th = resize_axis(Gm_th, (2*lmax - 1), mode='zero', axis=0)


# Second transform --- theta to m'
## Apply the periodic extension.
Gm_th = np.pad(Gm_th, [(0, 0),(0, Nt - 1)], mode='constant')
em = np.fft.ifftshift(np.arange(-(lmax-1), lmax))
Gm_th[:, Nt:] = ((-1.0)**(s + em) * Gm_th[:, 2*Nt-2 - np.arange(Nt, 2*Nt - 1)].T).T

Fmm = np.fft.fft(Gm_th, axis=1) / (2* Nt - 1)
Fmm = (Fmm.T * np.exp(-1j * em * dth)).T   # Phase offset (since thetas[0] != 0) *(only applies for ssht sampling)

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


## Finally, do the sum over Deltas

wig_d = dMat(lmax)

# Define index raveling for flm matrix

def unravel_lm(l, m):
    return l**2 + l +  m

def ravel_lm(ind):
    l = int(np.floor(np.sqrt(ind)))
    m = ind - l * (l + 1)

    return l, m


#import IPython; IPython.embed()
## TODO Define a better receptacle for flm.
flm = np.zeros((1+lmax)**2, dtype=complex)
offset_m = lmax - 1
for el in range(lmax + 1):
    for m in range(-(el-1), el+1):
        ind = unravel_lm(el, m)
        for mp in range(-el, el+1):
            flm[ind] += (-1)**(el+m) * np.sqrt((2*el + 1)/(4*np.pi)) * (1j)**(m+s) * wig_d[el, mp, m] * wig_d[el, -s, mp] * Gmm[m, mp]

import IPython; IPython.embed()
