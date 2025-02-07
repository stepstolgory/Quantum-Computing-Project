from .gate import Gate
import numpy as np

class H(Gate):
    """
    Represents the H (Hadamard) quantum gate (sparse matrix).

    This gate is represented by the 2x2 matrix:

        H = 1/sqrt(2) * [[1, 1],
                        [1, -1]]

    Attributes are inherited from the `Gate` class, with the matrix data, row and column indices, and sparse flag set for the identity gate.
    """
    def __init__(self):
        data = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2), -1 / np.sqrt(2)])
        row = np.array([0, 0, 1, 1])
        col = np.array([0, 1, 0, 1])
        sparse = True
        super().__init__(data, row, col, sparse)