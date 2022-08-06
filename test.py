from __future__ import division
from numba import cuda, float32
import numpy as np
import math

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16


@cuda.jit
def test(A, B):
    x, y = cuda.grid(2)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if 0 <= y + i < B.shape[0] and 0 <= x + j < B.shape[1]:
                cuda.atomic.add(B, (y + i, x+j), 1)
                cuda.atomic.add(A, (y + i, x+j), 1)


A = np.zeros((100, 100))

B = np.zeros((100, 100))

A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)


# Configure the blocks
threadsperblock = (TPB, TPB)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid = int(math.ceil(A.shape[0] / threadsperblock[0]))

# Start the kernel
test[blockspergrid, threadsperblock](A_global_mem, B_global_mem)
A = A_global_mem.copy_to_host()
B = B_global_mem.copy_to_host()

print(np.sum(A != B))
