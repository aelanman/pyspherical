# pyspherical

`pyspherical` implements the fast spin-weighted spherical harmonic transform methods of McEwan and Wiaux (2011) [1],
and evaluates Wigner little-d functions using the recursion relations of Trapani and Navaza (2006) [2]. Transforms are
supported for any spherical sampling pattern with equally-spaced samples of azimuth at each latitude (iso-latitude sampling).
Additional functions are provided to evaluate spin-weighted spherical harmonics at arbitrary positions.

These methods are implemented entirely in Python, taking advantage of numba jit compilation and numpy vector operations
for speed.

## Dependencies
* `numpy`
* `numba`
* `scipy`

## Installation

`pyspherical` may be installed by cloning the repository and running setup.py:
```
git clone https://github.com/aelanman/pyspherical.git
pip install .
```

## References

[1] McEwen, J. D., and Y. Wiaux. “A Novel Sampling Theorem on the Sphere.” IEEE Transactions on Signal Processing, vol. 59, no. 12, Dec. 2011, pp. 5876–87. arXiv.org, doi:10.1109/TSP.2011.2166394.

[2] S, Trapani, and Navaza J. “Calculation of Spherical Harmonics and Wigner d Functions by FFT. Applications to Fast Rotational Matching in Molecular Replacement and Implementation into AMoRe.” Acta Crystallographica Section A, vol. 62, no. 4, 2006, pp. 262–69. Wiley Online Library, doi:10.1107/S0108767306017478.