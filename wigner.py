
### Better -- This source has a direct recursion relation for the beta = pi/2 terms, which is what I want:
##   Eqns 9 - 12
##  http://scripts.iucr.org/cgi-bin/paper?S0108767306017478
##   Ref 41 of "A novel sampling theorem on the sphere"


import numpy as np
from numba import jit, int32


@jit(int32(int32,int32,int32),nopython=True)
def _tri_ravel(l, m1, m2):
    # Ravel indices for the 'stack of triangles' format.
    # m1 must be >= m2
    if m1 < m2 or m1 > l or m2 > l: 
        raise ValueError("Invalid indices")
        #raise ValueError('Invalid indices: {}, {}, {}'.format(l,m1,m2)) 
    base = l * (l + 1) * (l+2) // 6
    offset = (l - m1)*(l + 3 + m1) // 2 + m2
    ind = base + offset 
    return int(ind)

@jit(nopython=True)
def _dmat_eval(lmax, arr):
    # Evaluate the values of the Wigner-d matrices at pi/2.
    # arr = linear array, modified in-place
    arr[0] = 1.0
    for el in range(1, lmax+1):
        dll0 = -np.sqrt((2*el - 1)/float(2*el)) * arr[_tri_ravel(el - 1, el - 1, 0)]
        arr[_tri_ravel(el, el, 0)] = dll0
        for m2 in range(0, el+1):
            if m2 > 0:
                dllm = np.sqrt(
                        (el/2.) * (2* el - 1)/ ((el + m2)*(el + m2 - 1))
                        ) * arr[_tri_ravel(el - 1, el - 1, m2 - 1)]
                arr[_tri_ravel(el, el, m2)] = dllm
            for m1 in range(el-1, m2-1, -1):
                fac1 = (2 * m2)/np.sqrt( (el - m1) * (el + m1 + 1)) * arr[_tri_ravel(el, m1 + 1, m2)]
                fac2 = 0.0
                if (m1 + 2) <= el:
                    fac2 = np.sqrt( ( (el - m1 - 1) * (el + m1 + 2) )/ ( (el - m1) * (el + m1 + 1) ) ) * arr[_tri_ravel(el, m1 + 2, m2)]
                arr[_tri_ravel(el, m1, m2)] = fac1 - fac2


class dMat(object):
    """
    Describes a matrix of Deltas.
    Only stores values for m1, m2 >= 0.

    Other values are returned by symmetry.
    """
    ## TODO
    #       *Have shown that numba acceleration is helping.
    #       *Test against another package -- mathematica / spherical_functions 
    #       Handling slices?
    #       *Numba acceleration on recursive calc and indexing.
    #       *Accessing non-stored values --- using symmetry relations
    #       Cite the paper defining the recursive relations (https://arxiv.org/pdf/1110.6298.pdf)
    #       Checking recursive stability for large l -- need to compare against a different calculation.
    #       Limited storage case -- only store after an lmin and up to lmax.
    #       "unravel" method

    def __init__(self, lmax):
        arrsize = (1+lmax) * (2+lmax) * (3+lmax) // 6
        self._arr = np.empty(arrsize, dtype=np.float32)
        self.lmax = lmax
        self.size = arrsize
        self._eval()

    def _eval(self):
        _dmat_eval(self.lmax, self._arr)

    @classmethod
    def ravel_index(cls, l, m1, m2): 
        # Ravel indices for the 'stack of triangles' format.
        return _tri_ravel(l, m1, m2)

    def __getitem__(self, index):
        # Access stored elements, or use 
        # symmetry relations for non-stored elements.

        (l, m1, m2) = index
        fac = 1.0
        # flip negatives
        if m1 < 0:
            fac *= (-1)**(l - index[2])
            m1 *= -1
        if m2 < 0:
            fac *= (-1)**(l - index[1])
            m2 *= -1

        # reverse if wrong order
        if m1 < m2:
            fac *= (-1)**(m1 - m2)
            m1, m2 = m2, m1
        #print(index, m1, m2, fac)
        val = fac * self._arr[self.ravel_index(l, m1, m2)]
        if np.isnan(val):
            raise ValueError("Invalid entry in dmat")
        return val

    def __setitem__(self, index, value):
        (l, m1, m2) = index
        ind = self.ravel_index(l, m1, m2)
        self._arr[ind] = value


#TODO  Now -- Define the spherical harmonic transform!
#       > I've been able to compile the python wrapper on ssht (it's not in the path. Look to the .so file in the ssht dir.
#       > Can do comparable transforms here and there and compare.


## TODO Move this test stuff to a different file.


# Verify against another package

##import quaternion
##import spherical_functions as sf
##
##lmax = 40 
##dmat = dMat(lmax)
##import IPython; IPython.embed()
##import sys; sys.exit()
##dmat.eval()
##
##R = quaternion.from_euler_angles(0, np.pi/2., 0)
##for el in range(0,lmax+1):
##    for m1 in range(-el, el):
##        for m2 in range(-el, el):
##            sfval = sf.Wigner_D_element(R, el, m1, m2)
##            # These lines -- confirm the symmetry relations
##            #print(np.isclose(sfval, sf.Wigner_D_element(R, el, -m1, m2) * (-1)**(el- m2)))
##            #print(np.isclose(sfval, sf.Wigner_D_element(R, el, -m1, -m2) * (-1)**(m1- m2)))
##            #print(np.isclose(sfval, sf.Wigner_D_element(R, el, m1, -m2) * (-1)**(el- m1)))
##            check = np.isclose(sfval.real, dmat[el, m1, m2])
##            try:
##                assert check
##            except AssertionError:
##                import IPython; IPython.embed()
##                import sys; sys.exit()
##            assert np.isclose(sfval.imag, 0)
##
#import IPython; IPython.embed()
#import sys; sys.exit()
