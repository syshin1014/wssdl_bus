# coded by syshin
# to compute an unidirectional intersection area,
# which represents how much each box in 'boxes' is overlapped with boxes in 'query_boxes'

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def bbox_overlaps_ui(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef unsigned int k, n
    for n in range(N):
        box_area = (
            (boxes[n, 2] - boxes[n, 0] + 1) *
            (boxes[n, 3] - boxes[n, 1] + 1)
        )
        for k in range(K):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    overlaps[n, k] = iw * ih / box_area
                    
    return overlaps