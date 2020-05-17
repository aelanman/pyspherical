import numpy as np
import pytest

import pyspherical as pysh

"""
This will only run if NUMBA_DISABLE_JIT is in the environmental variables.
For automated testing, this needs to be run as a separate github action.

This allows coverage to see the contents of JIT functions.
"""

pytestmark = pytest.mark.skipif('not "NUMBA_DISABLE_JIT" in os.environ')


def test_delta_eval():
    # _dmat_eval
    # Check that values are equivalent for single and double.

    lmax = 20
    arr0 = np.empty(pysh.DeltaMatrix.estimate_array_size(0, lmax), dtype=np.float32)
    arr1 = np.empty(pysh.DeltaMatrix.estimate_array_size(0, lmax), dtype=np.float64)
    pysh.wigner._dmat_eval(lmax, arr0, lmin=0, lstart=None, arr0=None)

    pysh.wigner._dmat_eval(lmax, arr1, lmin=0, lstart=None, arr0=None)

    # Starting from an lmin
    lmin = 5
    arr2 = np.empty(pysh.DeltaMatrix.estimate_array_size(lmin, lmax), dtype=np.float32)
    base = pysh.utils.tri_base(lmin)
    bsize = pysh.utils.el_block_size(lmin)
    pysh.wigner._dmat_eval(lmax, arr2, lmin=lmin, lstart=lmin, arr0=arr1[base:base + bsize])

    assert np.allclose(arr0, arr1, atol=1e-4)
    assert np.allclose(arr1[pysh.utils.tri_base(lmin):], arr2)


@pytest.mark.parametrize('lmin', [0, 3, 7])
def test_indexing_and_sizes(lmin):

    lmax = 10
    tot = pysh.utils.tri_base(lmin)
    for el in range(lmin, lmax + 1):
        assert pysh.utils.tri_base(el) == tot
        tot += pysh.utils.el_block_size(el)
    assert pysh.DeltaMatrix.estimate_array_size(lmin, lmax) == tot - pysh.utils.tri_base(lmin)

    # Error for invalid index
    with pytest.raises(ValueError, match="Invalid indices"):
        pysh.utils.tri_ravel(0, 1, 0)

    # Check that tri_ravel indices cover the full array.
    inds = []
    for el in range(lmin, lmax + 1):
        for m1 in range(0, el + 1):
            for m2 in range(0, m1 + 1):
                inds.append(pysh.utils.tri_ravel(el, m1, m2))

    expected_size = pysh.DeltaMatrix.estimate_array_size(lmin, lmax)
    assert np.all(sorted(inds) == np.arange(expected_size) + pysh.utils.tri_base(lmin))


def test_ravel_lm():
    # Check that ravel_lm is consistent with unravel_lm:
    lmax = 10
    inds = np.arange(lmax**2)
    el, em = np.array([pysh.utils.ravel_lm(inds[ii]) for ii in range(lmax**2)]).T
    assert np.all(np.abs(em) <= el)
    assert np.all([inds[ii] == pysh.utils.unravel_lm(el[ii], em[ii]) for ii in range(lmax**2)])


def test_access_element():
    # Results of _access_element respect the same symmetries as checked in test_wigner.py
    lmax = 20
    arr0 = np.empty(pysh.DeltaMatrix.estimate_array_size(0, lmax), dtype=np.float32)
    arr2 = np.ones_like(arr0) * np.nan
    pysh.wigner._dmat_eval(lmax, arr0, lmin=0, lstart=None, arr0=None)

    _acc = pysh.wigner._access_element

    with pytest.raises(ValueError, match="Invalid entry in dmat"):
        _acc(2, 0, 0, arr2)

    for ll in range(lmax):
        for m in range(-ll, ll + 1):
            for mm in range(-ll, ll + 1):
                val1 = _acc(ll, m, mm, arr0)

                assert np.isclose(val1, (-1)**(m - mm) * _acc(ll, -m, -mm, arr0))
                assert np.isclose(val1, (-1)**(m - mm) * _acc(ll, mm, m, arr0))
                assert np.isclose(val1, _acc(ll, -mm, -m, arr0))
                assert np.isclose(
                    _acc(ll, m, mm, arr0), (-1)**(ll - mm) * _acc(ll, -m, mm, arr0), atol=1e-10)


def test_transform():
    # To cover the compiled methods in transforms.py
    lmax = 10
    theta, phi = pysh.utils.get_grid_sampling(lmax=lmax)

    amp = 20
    dat = np.ones((theta.size, phi.size)) * amp

    flm = pysh.forward_transform(dat, phi, theta, lmax)
    res = pysh.inverse_transform(flm, phi, theta, lmax)
    assert np.isclose(flm[0], amp * np.sqrt(4 * np.pi))
    assert np.allclose(res, amp, atol=1e-4)


def test_wigner_d_matrix_el():
    # Check the _get_matrix_elements function

    dmat = pysh.DeltaMatrix(6)

    el = 5
    m1 = 3
    m2 = 2

    test = []
    for mp in range(0, el + 1):
        test.append(dmat[el, mp, m1] * dmat[el, mp, m2])

    test = np.array(test)

    comp = np.empty(test.size, dtype=np.float32)
    pysh.wigner._get_matrix_elements(el, m1, m2, dmat._arr, comp)

    assert np.allclose(comp, test)
