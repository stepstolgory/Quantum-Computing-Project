from .gate import Gate
import numpy as np

class Y(Gate):
    """
    Represents the Y quantum gate (sparse matrix).

    This gate is represented by the 2x2 matrix:

        Y = [[0, -i],
             [i, 0]]

    Attributes are inherited from the `Gate` class, with the matrix data, row and column indices, and sparse flag set for the identity gate.
    """
    def __init__(self):
        data = np.array([-1j, 1j])
        row = np.array([0, 1])
        col = np.array([1, 0])
        sparse = True
        super().__init__(data, row, col, sparse)