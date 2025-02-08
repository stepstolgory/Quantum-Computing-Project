import numpy as np
import scipy.sparse as sps
from quantum_computing_project.operations import Operations

class TestOperations:

    @classmethod
    def setup_class(cls):
        """Sets up sparse and non-sparse matrices for testing."""
        cls.zero = sps.coo_matrix(([1], ([0], [0])), shape=(2, 1))
        cls.one = sps.coo_matrix(([1], ([1], [0])), shape=(2, 1))

    def test_tensor(self):
        assert False


    def test_sparse_tensor(self):
        assert False


    def test_power_tensor(self):
        assert False


    def test_sparse_power_tensor(self):
        assert False
