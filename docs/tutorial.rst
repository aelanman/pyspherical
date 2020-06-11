Tutorial
========

.. # H1: =, H2: -, H3: ^, H4: ~, H5: ", H6: #


Sampling
--------

The function ``get_grid_sampling`` can be used to get a uniform grid sampling in Latitude/Longitude.

::
    >>> from pyspherical import get_grid_sampling

    # Given lmax, this returns lmax samples in theta and (2 * lmax - 1) samples in phi.
    # This is the default sampling of the McEwen + Wiaux paper (MW).
    >>> lmax = 10
    >>> theta, phi = get_grid_sampling(lmax)
    >>> print(theta.shape, phi.shape)
    (10,) (19,)

    # Alternatively, Nt and Nf can be used to set certain numbers of thetas/phis.
    >>> theta, phi = get_grid_sampling(Nt=20, Nf=50)
    >>> print(theta.shape, phi.shape)
    (20,) (50,)


DeltaMatrix
-----------

The transformations and function evaluations all take advantage of a cached DeltaMatrix, which gives the
values of the Wigner-d (little-d) function at an angle of π/2.

::
    >>> from pyspherical.wigner import DeltaMatrix
    >>> dmat = DeltaMatrix(lmax=5)
    # Elements (el, m1, m2) of the delta matrix can be accessed by index:
    #   dmat[el, m1, m2]
    >>> dmat[2, -1, 1]
    0.5
    # Data are only saved for 0 <= m1 < m2. Additional entries are given by symmetry relations.
    # Slicing is not yet available.
    # The DeltaMatrix can also be initialized with an lmin value:
    >>> dmat2 = DeltaMatrix(lmin=4, lmax=10)
    >>> dmat2[1, 0, 1]
    ValueError: l < lmin. Need to re-evaluate delta matrix.


Evaluation
----------

Functions for directly evaluating the spin-weighted spherical harmonics and Wigner-d functions are provided at the top level of the module.

::
    >>> from pyspherical import spin_spherical_harmonic, wigner_d
    >>> from pyspherical.wigner import DeltaMatrix
    >>> import numpy as np
    >>> phi, theta = 0, np.pi/2
    >>> el = 5
    >>> em = 4
    >>> spin = 1
    >>> spin_spherical_harmonic(spin, el, em, theta, phi)
    (0.26796685259615366+0j)

    >>> m1, m2 = 1, 3
    >>> wigner_d(el, m1, m2, theta=np.pi/2)
    (-0.2025231+0j)
    # Note that the above is equivalent to DeltaMatrix[el, m1, m2]
    >>> dmat = DeltaMatrix(lmax=6)
    >>> dmat[el, m1, m2]
    -0.2025231
    # By default, all functions and DeltaMatrices are evaluated to single precision.
    # Double precision can be used by setting the keyword double_prec=True.


Transforms
----------

The functions ``forward_transform`` and ``inverse_transform`` perform the discrete transforms from a set of samples on a sphere to a set of harmonic components, and back, respectively.

::
    >>> from pyspherical import forward_transform, inverse_transform, get_grid_sampling, spin_spherical_harmonic
    >>> from pyspherical.utils import unravel_lm, ravel_lm
    >>> import numpy as np

    >>> lmax = 20   # Maximum multipole moment.
    >>> theta, phi = get_grid_sampling(lmax)
    >>> gtheta, gphi = np.meshgrid(theta, phi)

    # Set up some test data
    # This test is just the SH function with spin 1, l=2, and m=0.
    >>> dat = spin_spherical_harmonic(1, 2, 0, gtheta, gphi)

    # ------- FORWARD -------
    >>> flm = forward_transform(dat, phi, theta, lmax, spin=1)

    # The transformations can be run with either:
    #   [dat.shape == (phi.size, theta.size)] or [dat.shape == phi.shape == theta.size]
    >>> flm2 = forward_transform(dat, gphi, gtheta, lmax, spin=1)
    >>> (flm2 == flm).allclose()
    True

    # The flm has shape (lmax-1)**2, with indices ordered as:
    #        ind = 0  1  2  3  4  5  6  7  8
    #        l  = 0  1  1  1  2  2  2  2  2
    #        m  = 0 -1  0  1 -2 -1  0  1  2
    # l, m = ravel_lm(ind)
    # ind = unravel_lm(l, m)

    >>> peak_ind = np.argmax(flm.real)
    >>> print(ravel_lm(peak_ind))
    (2, 0)
    # Matches the amplitude of the peak inserted.
    # (See scripts/example_1.py for a more complete example).

    # ------- INVERSE -------
    # Now the inverse transform:
    >>> dat2 = inverse_transform(flm, phi, theta, lmax, spin=1)
    >>> np.allclose(dat, dat2)
    True


Caching
-------
