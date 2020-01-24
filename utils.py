
import numpy as np

def resize_axis(arr, size, mode='zero', axis=0):
    """
    Resize an axis of the array.

    Parameters
    ----------
    arr : ndarray
        Array to resize
    size : int
        New size for the axis.
    mode : str
        If the new array is zero-padded, where to put the old data:
            * 'zero' : Put zeros in the middle of the axis (center data on zero)
            * 'start' : Put zeros at the end of the axis.
            * 'center': Evenly fill data on both sides.
    axis : int
        Which axis to resize
    """

    shape = list(arr.shape)
    shape[axis] = size

    new = np.zeros(tuple(shape), dtype=arr.dtype)
    
    oldodd = arr.shape[axis] % 2 == 1
    newodd = size % 2 == 1

    L = arr.shape[axis]
    if oldodd:
        center = (L - 1) // 2   # This stays on the left.
    else:
        center = L//2
    if newodd:
        newcent = (size - 1) // 2
    else:
        newcent = size//2

    _arr = np.swapaxes(arr, axis, 0)
    _new = np.swapaxes(new, axis, 0)

    trunc = size < L
    # Better -- Roll the chosen spot to the end, pad there, then roll back.

    if mode == 'zero':
        limit = np.min([center, newcent])
        if oldodd:
            _new[:limit+1, ...] = _arr[:limit+1, ...]
            _new[-1:-limit-1:-1, ...] = _arr[-1:-limit-1:-1, ...]
        else:
            _new[:limit, ...] = _arr[:limit, ...]
            _new[-1:-limit-1:-1, ...] = _arr[-1:-limit-1:-1, ...]

    elif mode == 'start':
        _new[:L, ...] = _arr[:L, ...]
    elif mode == 'center':
        if size <= L:
            # Truncating
            base = int(np.floor((L - size)/2))
            print(_arr)
            _new[:, ...] = _arr[base:base+size, ...]
        else:
            base = int(np.floor((size - L)/2))
            _new[base:base+L, ...] = _arr[:, ...]

    new = np.swapaxes(_new, 0, axis)

    return new

if __name__ == '__main__':
    import sys
    arr = np.arange(int(sys.argv[1]), dtype=int) + 1
    #test = resize_axis(arr, int(sys.argv[2]), 'zero')
    test = resize_axis(arr, int(sys.argv[2]), 'center')
    print(test, test.shape, np.sum(test))
#arr = np.repeat(np.arange(int(sys.argv[1]))[None, :], 3, axis=0)
#test = resize_axis(arr, int(sys.argv[2]), axis=1)
#print(test)
#print(test.shape, arr.shape)
#
#N = int(sys.argv[1])
#testarr = np.zeros((3, int(sys.argv[2])), dtype=int)
#testarr[:, :(N-1)//2] = arr[:, :(N-1)//2]
#testarr[:, -1:-(N-1)//2:-1] = arr[:, -1:-(N-1)//2:-1]
#
#import IPython; IPython.embed()
