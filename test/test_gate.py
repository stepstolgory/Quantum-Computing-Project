import numpy as np
from quantum_computing_project.gate import Gate

class TestGate:
    """
    Test suite for the gate package, testing various cases for real and complex gates
    (both dense and sparse) to ensure correct functionality.
    """

    def test_real(self):
        """
        Test case for creating a real-valued gate (non-sparse).
        
        This test checks if real values are correctly assigned to the I operator gate. 

        Asserts:
            - The created gate is a 2x2 identity matrix.
        """
        data = np.array([1, 1])
        row = np.array([0,1])
        col = np.array([0,1])
        sparse = False
        i = Gate(data, row, col, sparse).gate
        expected_i = np.array([[1, 0],[0, 1]])
        np.testing.assert_array_equal(i, expected_i)

    def test_imaginary(self):
        """
        Test case for creating a gate with complex values (non-sparse).
        This test checks if complex values are correctly assigned to the S operator gate. 

        Asserts:
            - The created gate has complex values and matches the expected output.
        """
        data = np.array([1, 1j])
        row = np.array([0,1])
        col = np.array([0,1])
        sparse = False
        s = Gate(data, row, col, sparse).gate
        expected_s =  np.array([[1, 0],[0, 1j]])
        np.testing.assert_array_equal(s, expected_s)

    def test_real_sparse(self):
        """
        Test case for creating a real-valued gate (sparse).
        
        This test checks if real values are correctly assigned to the I operator gate. 

        Asserts:
            - The created gate is a 2x2 identity matrix.
        """
        data = np.array([1, 1])
        row = np.array([0,1])
        col = np.array([0,1])
        sparse = True
        i = Gate(data, row, col, sparse).gate
        expected_i = np.array([[1, 0],[0, 1]])
        np.testing.assert_array_equal(i.toarray(), expected_i)

    def test_imaginary_sparse(self):
        """
        Test case for creating a gate with complex values (sparse).
        This test checks if complex values are correctly assigned to the S operator gate. 

        Asserts:
            - The created gate has complex values and matches the expected output.
        """
        data = np.array([1, 1j])
        row = np.array([0,1])
        col = np.array([0,1])
        sparse = True
        s = Gate(data, row, col, sparse).gate
        expected_s =  np.array([[1, 0],[0, 1j]])
        np.testing.assert_array_equal(s.toarray(), expected_s)

    