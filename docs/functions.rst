Function Documentation
======================

Transforms
----------
.. automodule:: pyspherical.transforms
   :members:

Evaluation
----------
.. automodule:: pyspherical.wigner
   :members: wigner_D, spin_spherical_harmonic, spin_spharm_goldberg

Utilities
---------

.. automodule:: pyspherical.utils
   :members:

Caching
-------

The methods used to calculate spherical harmonic transforms and Wigner-D functions use an array of pre-computed
values of the Wigner-d functions at θ = π / 2, called a ``DeltaMatrix``. This matrix is computed for a particular
range of :math:`l` values, respecting a fixed limit of memory usage. These functions control this caching behavior.

.. automodule:: pyspherical.wigner
   :noindex:
   :members: get_cached_dmat, clear_cached_dmat, set_cache_mem_limit, get_cache_details
