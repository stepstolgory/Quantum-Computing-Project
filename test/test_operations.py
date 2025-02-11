import numpy as np
import scipy.sparse as sps
from quantum_computing_project.operations import Operations
from quantum_computing_project import constants

class TestOperations:
    """
    Test suite for verifying tensor operations on sparse and dense matrices.
    """
    @classmethod
    def setup_class(cls):
        """Sets up sparse and non-sparse matrices for testing."""
        cls.s_sparse = constants.S.gate
        cls.t_sparse = constants.T.gate
        cls.s_arr = np.array([[1, 0], [0, 1j]])
        cls.t_arr = np.array([[1, 0], [0, 1/np.sqrt(2)+1j/np.sqrt(2)]])

    def test_tensor(self):
        """
        Test case for computing the tensor product of two dense matrices.

        Asserts:
            - The computed tensor product is nearly equal (accounts for floats)
            to the expected matrix.
        """
        result = Operations.tensor(self.s_arr, self.t_arr)
        expected = np.array([[1, 0, 0, 0],
                                    [0, 1/np.sqrt(2)+1j/np.sqrt(2), 0, 0],
                                    [0, 0, 1j, 0],
                                    [0, 0, 0, 1j/np.sqrt(2)-1/np.sqrt(2)]])
        np.testing.assert_array_almost_equal(result, expected)


    def test_sparse_tensor(self):
        """
        Test case for computing the tensor product of two sparse matrices.

        Asserts:
            - The computed tensor product is nearly equal (accounts for floats)
            to the expected matrix.
        """
        result = Operations.sparse_tensor(self.s_sparse, self.t_sparse)
        expected_data = np.array([1, 1/np.sqrt(2)+1j/np.sqrt(2), 1j, -1/np.sqrt(2)+1j/np.sqrt(2)])
        expected_row = np.array([0, 1, 2, 3])
        expected_col = np.array([0, 1, 2, 3])
        expected = sps.coo_matrix((expected_data, (expected_row, expected_col)))
        np.testing.assert_array_almost_equal(result.toarray(), expected.toarray())


    def test_power_tensor(self):
        """
        Test case for computing repeated tensor products of a dense matrix.

        Asserts:
            - The computed power tensor product is nearly equal (accounts for floats)
            to the expected matrix.
        """
        result = Operations.power_tensor(self.s_arr, 2)
        expected = np.array([[1, 0, 0, 0],
                                    [0, 1j, 0, 0],
                                    [0, 0, 1j, 0],
                                    [0, 0, 0, -1]])
        np.testing.assert_array_almost_equal(result, expected)


    def test_sparse_power_tensor(self):
        """
        Test case for computing repeated tensor products of a sparse matrix.

        Asserts:
            - The computed power tensor product is nearly equal (accounts for floats)
            to the expected matrix.
        """
        result = Operations.sparse_power_tensor(self.s_sparse, 2)
        expected_data = np.array([1, 1j, 1j, -1])
        expected_row = np.array([0, 1, 2, 3])
        expected_col = np.array([0, 1, 2, 3])
        expected = sps.coo_matrix((expected_data, (expected_row, expected_col)))
        np.testing.assert_array_almost_equal(result.toarray(), expected.toarray())
