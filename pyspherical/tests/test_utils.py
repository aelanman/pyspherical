
import numpy as np
import pytest

import pyspherical as pysh


def test_get_mw_grid():
    # Check that different methods fo getting an MW sampling grid are consistent and correct.
    lmax = 10
    Nt = lmax
    Nf = 2 * lmax - 1

    with pytest.raises(ValueError, match="Need to provide lmax"):
        th, ph = pysh.utils.get_grid_sampling()

    th, ph = pysh.utils.get_grid_sampling(lmax)
    assert th.size == Nt
    assert ph.size == Nf


@pytest.mark.parametrize('Nitems', [10, 11])
@pytest.mark.parametrize('Nnew', [5, 6, 10, 13, 14])
def test_resize_axis(Nitems, Nnew):
    # Zero-pad / truncate of arrays.
    #   Check that:
    #       * Data end up in the right places, with nothing lost, when zero-padding.
    #       * The right parts of the array are kept when truncating.

    arr0 = np.arange(Nitems) + 1
    arr1 = pysh.utils.resize_axis(arr0, Nnew, mode='zero', axis=0)
    arr2 = pysh.utils.resize_axis(arr0, Nnew, mode='start', axis=0)
    arr3 = pysh.utils.resize_axis(arr0, Nnew, mode='center', axis=0)

    newodd = Nnew % 2 == 1
    limit = min(Nnew, Nitems) // 2

    # "start" mode -- either truncate to the start of the array, or
    # pad with zeros after arr0.
    assert (arr2[:limit] == arr0[:limit]).all()
    if newodd:
        # "zero" mode -- array should match original, with zeros in the middle.
        assert (arr0[:limit] == arr1[:limit]).all()
        assert (arr0[-limit + 1:] == arr1[-limit + 1:]).all()
    else:
        assert (arr0[:limit] == arr1[:limit]).all()
        assert (arr0[-limit:] == arr1[-limit:]).all()

    diff = Nnew - Nitems
    if diff < 0:
        trunced = arr0.copy().tolist()
        for ii in range(-diff):
            if ii % 2 == 1:
                del trunced[0]
            else:
                del trunced[-1]
        assert trunced == arr3.tolist()
    else:
        padded = arr3.copy().tolist()
        for ii in range(diff):
            if ii % 2 == 1:
                del padded[0]
            else:
                del padded[-1]
        assert padded == arr0.tolist()

    # Now check that the sizes are correct and the original data is kept
    # in the right order with zero-padding.
    for arr in [arr1, arr2, arr3]:
        assert arr.size == Nnew
        if Nnew >= Nitems:
            assert (arr[arr > 0] == arr0).all()
