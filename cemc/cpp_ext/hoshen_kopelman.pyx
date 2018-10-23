# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def hoshen_kopelman(np.ndarray[np.uint8_t, ndim=3] matrix):
    """Run the Hoshen-Kopelman algorithm"""
    cdef np.ndarray[np.uint8_t, ndim=3] clusters = np.zeros_like(matrix)
    cdef int largest_label = 0
    cdef int nx = matrix.shape[0]
    cdef int ny = matrix.shape[1]
    cdef int nz = matrix.shape[2]
    cdef np.ndarray[np.uint8_t, ndim=1] labels = np.zeros(nx*ny*nz/2, dtype=np.uint8)
    cdef int left = 0
    cdef int depth = 0
    cdef int above = 0

    labels[0] = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if matrix[ix, iy, iz] > 0:
                    # This cell is occupied
                    if ix == 0:
                        left = clusters[nx-1, iy, iz]
                    else:
                        left = clusters[ix-1, iy, iz]
                    
                    if iy == 0:
                        above = clusters[ix, ny-1, iz]
                    else:
                        above = clusters[ix, iy-1, iz]
                    
                    if iz == 0:
                        depth = clusters[ix, iy, nz-1]
                    else:
                        depth = clusters[ix, iy, iz-1]

                    if (left == 0) and (depth == 0) and (above == 0):
                        # No neighbours
                        labels[0] += 1
                        clusters[ix, iy, iz] = labels[0]
                        labels[labels[0]] = labels[0]
                    elif (left != 0) and (depth == 0) and (above == 0):
                        # One neighbour to the left
                        clusters[ix, iy, iz] = find(clusters, left)
                    elif (left == 0) and (depth != 0) and (above == 0):
                        # One neighbour in the depth
                        clusters[ix, iy, iz] = find(clusters, depth)
                    elif (left == 0) and (depth == 0) and (above != 0):
                        # One neighbour above
                        clusters[ix, iy, iz] = find(clusters, above)
                    elif (left != 0) and (depth != 0) and (above == 0):
                        # Neighbours to the left and in depth
                        union(clusters, left, depth)
                        clusters[ix, iy, iz] = find(clusters, left)
                    elif (left != 0) and (depth == 0) and (above != 0):
                        # Neighbours to the left and above
                        union(clusters, left, above)
                        clusters[ix, iy, iz] = find(clusters, left)
                    elif (left == 0) and (depth != 0) and (above != 0):
                        # Neighbours in depth and above
                        union(clusters, depth, above)
                        clusters[ix, iy, iz] = find(clusters, depth)
                    else:
                        # Neighbours in all directions
                        union(labels, left, depth)
                        union(labels, left, above)
                        clusters[ix, iy, iz] = find(clusters, left)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int find(np.ndarray[np.uint8_t, ndim=1] labels, int label):
    cdef int temp_label = label
    while (temp_label != labels[temp_label]):
        temp_label = labels[temp_label]

    cdef int temp_lab2 = 0
    while (labels[label] != label):
        temp_lab2 = labels[label]
        labels[label] = temp_label
        label = temp_lab2
    return temp_label

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void union(np.ndarray[np.uint8_t, ndim=1] labels, int lab1, int lab2):
    labels[find(labels, lab1)] = labels[find(labels, lab2)]
