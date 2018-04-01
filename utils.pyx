import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport round, sqrt

cdef inline float int_max(float a, float b): return a if a >= b else b
cdef inline float int_min(float a, float b): return a if a <= b else b

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def get_heat_maps(float cx,float cy,int vis,int stride,int gridx,int gridy,float sigma):
    cdef float start = stride/2.0 - 0.5
    cdef int gx, gy
    cdef float x, y, d2, exponent
    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros((gridy, gridx), dtype=np.float32)
    if vis==3:
        return result
    for gy in range(gridy):
        for gx in range(gridx):
            x = start + gx * stride
            y = start + gy * stride
            d2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
            result[gy, gx] = -d2/sigma/sigma
    result = np.exp(result)
    return result

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def get_vec_maps(double cx1, double cy1, double cx2, double cy2, int vis1, int vis2, int gridx,
                int gridy, int threshold, np.ndarray[np.float32_t, ndim=2] count):
    cx1 /=8
    cx2 /=8
    cy1 /=8
    cy2 /=8
    cdef np.ndarray[np.float32_t, ndim=3] result=np.zeros((gridy, gridx, 2), dtype=np.float32)
    if vis1==3 or vis2==3:
        return result
    cdef int minx = int(int_max(round(int_min(cx1, cx2)-threshold), 0))
    cdef int maxx = int(int_min(round(int_max(cx1, cx2)+threshold), gridx))
    cdef int miny = int(int_max(round(int_min(cy1, cy2)-threshold), 0))
    cdef int maxy = int(int_min(round(int_max(cy1, cy2)+threshold), gridy))
    cdef double bcx = cx1-cx2
    cdef double bcy = cy1-cy2
    cdef double normbc = sqrt(bcx**2+bcy**2)
    bcx /= normbc
    bcy /= normbc
    cdef float dist
    cdef double b2x, b2y
    cdef int x, y
    for y in range(miny, maxy):
        for x in range(minx, maxx):
            b2x = x-cx2
            b2y = y-cy2
            dist = abs(b2x*bcy-b2y*bcx)
            if dist <= threshold:
                count[y, x] += 1
                result[y, x, 0] = bcx
                result[y, x, 1] = bcy
    return result