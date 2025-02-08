import numpy as np
import scipy.sparse as sps

class Gate:
    """
    A class that represents a quantum gate.

    Attributes:
        gate (np.array or sps.coo_matrix): Matrix representing quantum gate.
        data (np.array): Array containing non-zero data values in the matrix.
        row (np.array): Array containing row indices where the non-zero values are placed.
        col (np.array): Array containing column indices where the non-zero values are placed.
        sparse (bool): Signals whether the matrix should be implemented as sparse.
    
    """
    def __init__(self, data, row, col, sparse):
        """
        Constructs necessary attributes for a quantum gate.

        Args:
            data (np.array): Array containing non-zero data values in the matrix.
            row (np.array): Array containing row indices where the non-zero values are placed.
            col (np.array): Array containing column indices where the non-zero values are placed.
            sparse (bool): Signals whether the matrix should be implemented as sparse.
        """
        self.sparse = sparse
        self.data = data
        self.row = row
        self.col = col

        if self.sparse:
            self.gate = sps.coo_matrix((self.data, (self.row, self.col)))
        else:
            self.gate = np.zeros((self.row.max() + 1, self.col.max() + 1), dtype=np.complex128)
            for i in range(data.size):
                self.gate[self.row[i], self.col[i]] = self.data[i]
