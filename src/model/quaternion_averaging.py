"""
Refer https://ntrs.nasa.gov/citations/20070017872

Code taken from https://github.com/christophhagen/averaging-quaternions
"""

import numpy as np

# Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
# The quaternions are arranged as (w,x,y,z), with w being the scalar
# The result will be the average quaternion of the input. Note that the signs
# of the output quaternion can be reversed, since q and -q describe the same orientation
def average_quaternions(Q: np.ndarray):
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros((4, 4))

    for i in range(0, M):
        q = Q[i, :]
        # multiply q with its transposed version q' and add A
        A = np.outer(q, q) + A

    # scale
    A = (1.0 / M) * A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:, eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:, 0].ravel())
