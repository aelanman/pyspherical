"""
An example of using spherical harmonic transforms and evaluation.

Generate a linear combination of spherical harmonics and
plot the transform, showing expected peaks.

"""


import numpy as np
import matplotlib.pyplot as pl

import pyspherical as pysh

lmax = 10
spin = 1

Npeaks = 7

Ntheta = 21
Nphi = 31

# Isolatitude and isolongitude spherical sampling.
theta, phi = pysh.utils.get_grid_sampling(Nt=Ntheta, Nf=Nphi)

# Meshgrid for function evaluation.
gtheta, gphi = np.meshgrid(theta, phi)

# Choose (el, em) harmonic modes to include.
peak_els = np.random.choice(np.arange(spin + 1, lmax - 1), Npeaks, replace=False)
peak_ems = np.array([np.random.randint(-el, el + 1) for el in peak_els])
peak_amps = np.random.uniform(10, 20, Npeaks)
dat = np.zeros(gtheta.shape, dtype=complex)
for ii in range(Npeaks):
    em = peak_ems[ii]
    el = peak_els[ii]
    dat += peak_amps[ii]\
        * pysh.spin_spherical_harmonic(el, em, spin, gtheta, gphi, lmax=lmax)

# Perform forward transform from data to harmonic components.

flm = pysh.forward_transform(dat, phi, theta, lmax, spin=spin)

# Plot the data, the flm, and the expected peak locations.
fig, (ax0, ax1) = pl.subplots(1, 2)

im = ax0.pcolormesh(np.degrees(theta), np.degrees(phi), dat.real)
pl.colorbar(im, ax=ax0)

ax1.plot(flm.real, label='Result')

# Get index in flm array from (el, em)
indices = [pysh.unravel_lm(peak_els[ii], peak_ems[ii]) for ii in range(Npeaks)]
ax1.scatter(indices, peak_amps, marker='o', color='r', label='Expected')

fig.suptitle(f"Spin {spin}", fontsize=14)
ax0.set_title("Re[Data]")
ax0.set_xlabel(r'$\theta$ [deg]')
ax0.set_ylabel(r'$\phi$ [deg]')

ax1.set_title(r"$_sf_{lm}$")
ax1.set_xlabel(r'index')
ax1.set_ylabel(r'Amplitude')
ax1.legend()
pl.tight_layout()
pl.show()
