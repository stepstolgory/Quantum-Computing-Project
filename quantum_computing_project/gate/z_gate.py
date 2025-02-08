from .gate import Gate
import numpy as np

class Z(Gate):
    """
    Represents the Z quantum gate (sparse matrix).

    This gate is represented by the 2x2 matrix:

        Z = [[1, 0],
             [0, -1]]

    Attributes are inherited from the `Gate` class, with the matrix data, row and column indices, and sparse flag set for the identity gate.
    """
    def __init__(self):
        data = np.array([1, -1])
        row = np.array([0, 1])
        col = np.array([0, 1])
        sparse = True
        super().__init__(data, row, col, sparse)