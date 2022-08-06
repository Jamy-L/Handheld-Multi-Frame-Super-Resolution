import numpy as np


A = np.array([[0, 1, 0, 1, 0, 1],
              [1, 2, 1, 2, 1, 2],
              [0, 1, 0, 1, 0, 1],
              [1, 2, 1, 2, 1, 2],
              [0, 2, 0, 1, 0, 1]])

B = np.zeros((A.shape[0]*2, A.shape[1]*2))
