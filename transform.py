
# Experiment script for the integrals over theta and phi

import numpy as np
from utils import resize_axis



lmax = 100   # Maximum el
Nt = 501   # Number of samples in theta (must be odd)
Nf = 401 # Samples in phi

# Define samples
theta = np.linspace(0, np.pi, Nt, endpoint=True)
phi = np.linspace(0, 2*np.pi, Nf, endpoint=False)

# Data, shape (Nf , Nt)

dat = np.random.uniform(0, 1, (Nf, Nt))


## First transform --- FFT phi to m, then zero pad or truncate to (2*lmax - 1)
Gm_th = np.fft.fft(dat, axis=0) * 2*np.pi / Nf

# Check that entries are in the right places.
#    Have checked -- np.fft.fftfreq(N, d=M/N) == resize_axis(np.fft.fftfreq(M), N)
#                    for the corresponding entries.
Gm_th = resize_axis(Gm_th, (2*lmax - 1), mode='zero', axis=0)


## Second transform --- theta to m'

# First, apply the periodic extension.
Gm_th = np.pad(Gm_th, [(0,0),(0,Nt - 2)], mode='constant')
em = np.arange(-(lmax-1), lmax)
s = 0   # Spin, for now
Gm_th[:, Nt:] = ((-1.0)**(s + em) * Gm_th[:, 2*Nt-2 - np.arange(Nt, 2*Nt - 2)].T).T

Fmm = np.fft.fft(Gm_th, axis=1) / (2 * np.pi * (2* Nt -  2))

# Truncate/zero-pad the m' axis, then transform over m'
padFmm = resize_axis(Fmm, (2*lmax - 1), axis=1, mode='zero')
padFmm = np.fft.fftshift(padFmm, axes=1)

# Get the weights and ifft them:
def weight(em):
    if np.abs(em) == 1:
        return np.sign(em) * np.pi * 1j / 2.
    if em % 2 == 1:
        return 0
    return 2/(1 - em**2)

em_ext = range(-2*(lmax - 1), 2 * (lmax - 1)+1)
ite = (weight(mm) for mm in em_ext)
weights = np.fromiter(ite, dtype=complex)

def do_conv(a):
    # The valid option only works with these Fourier-transformed
    # quantities if they have been FFT-shifted, such that the 0 mode
    # is in the middle of the array.
    return np.convolve(weights, a, mode='valid')

Gmm = 2 * np.pi * np.apply_along_axis(do_conv, 1, padFmm)

## Test -- trapezoidal sum.
#    This comparison is a little sketchy, but in a controlled case
#    it seems to verify that the convolution with weights is equivalent
#    to a trapzeoidal sum 
#em = np.array(range(-(lmax - 1), lmax ))
#Gm_th = np.fft.fft(dat, axis=0) * 2*np.pi / Nf
#Gm_th = resize_axis(Gm_th, (2*lmax - 1), mode='zero', axis=0)
#fours = np.exp(-1j * em[None,:] * theta[:,None])
#import scipy.integrate as sint
#integrand = ((np.sin(theta) * Gm_th)[:,:,None]* fours)        # SLOW
#Gmm_test = sint.trapz(integrand, x=theta , axis=1)



## Finally, do the sum over Deltas


import IPython; IPython.embed()
