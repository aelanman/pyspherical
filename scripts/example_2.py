"""Plot all spherical harmonic functions for the first 3 el modes."""

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import cm
import matplotlib.gridspec as gridspec

import pyspherical as pysh

spin = 0

lmax = 3

Ntheta = 21
Nphi = 31

theta = np.linspace(0, np.pi, Ntheta)
phi = np.linspace(0, 2 * np.pi, Nphi)

# Meshgrid for function evaluation.
gtheta, gphi = np.meshgrid(theta, phi)

xx = np.sin(gtheta) * np.cos(gphi)
yy = np.sin(gtheta) * np.sin(gphi)
zz = np.cos(gtheta)

fig = pl.figure(figsize=pl.figaspect((lmax + 1) / (2 * lmax + 1)))

gs = gridspec.GridSpec(lmax + 1, 2 * lmax + 1)

for el in range(0, lmax + 1):
    for em in range(-el, el + 1):
        ci = lmax + em
        ax = fig.add_subplot(gs[el, ci], projection='3d')

        sph = pysh.wigner.spin_spherical_harmonic(
            el, em, spin, gtheta, gphi, lmax=lmax)
        cmin, cmax = sph.real.min(), sph.real.max()
        if cmin == cmax:
            sph = 0.5 * np.ones_like(sph, dtype=complex)
        else:
            sph = (sph - cmin) / (cmax - cmin)
        ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,
                        facecolors=cm.plasma(sph.real))
        ax.set_axis_off()
        pos = gs[el, ci].get_position(fig)

        if el == lmax:
            pos = gs[el, ci].get_position(fig)
            fig.text((pos.x0 + pos.x1) / 2, pos.y0, r'$m = $' + str(em))
        if em == 0:
            ax.set_title(r'$\ell = $' + str(el))

pl.show()
