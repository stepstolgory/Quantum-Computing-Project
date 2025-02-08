import numpy as np
from quantum_computing_project.gate import Gate, I, X, Y, Z, S, H, T

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
        np.testing.assert_array_equal(i, expected_i)

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
        np.testing.assert_array_equal(s, expected_s)

    def test_i_gate(self):
        """
        Test case for creating an I gate.

        Asserts:
            - The created gate is the expected sparse matrix.
        """
        i = I().gate
        expected_i = np.array([[1, 0],[0, 1]])
        np.testing.assert_array_equal(i, expected_i)

    def test_s_gate(self):
        """
        Test case for creating an S gate.

        Asserts:
            - The created gate is the expected sparse matrix.
        """
        s = S().gate
        expected_s = np.array([[1, 0],[0, 1j]])
        np.testing.assert_array_equal(s, expected_s)

    def test_x_gate(self):
        """
        Test case for creating an X gate.

        Asserts:
            - The created gate is the expected sparse matrix.
        """
        x = X().gate
        expected_x = np.array([[0, 1],[1, 0]])
        np.testing.assert_array_equal(x, expected_x)

    def test_y_gate(self):
        """
        Test case for creating a Y gate.

        Asserts:
            - The created gate is the expected sparse matrix.
        """
        y = Y().gate
        expected_y = np.array([[0, -1j],[1j, 0]])
        np.testing.assert_array_equal(y, expected_y)

    def test_z_gate(self):
        """
        Test case for creating a Z gate.

        Asserts:
            - The created gate is the expected sparse matrix.
        """
        z = Z().gate
        expected_z = np.array([[1, 0],[0, -1]])
        np.testing.assert_array_equal(z, expected_z)

    def test_h_gate(self):
        """
        Test case for creating a H (Hadamard) gate.

        Asserts:
            - The created gate is the expected sparse matrix.
        """
        h = H().gate
        expected_h = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)],[1 / np.sqrt(2), -1 / np.sqrt(2)]])
        np.testing.assert_array_equal(h, expected_h)

    def test_t_gate(self):
        """
        Test case for creating a T gate.

        Asserts:
            - The created gate is the expected sparse matrix.
        """
        t = T().gate
        expected_t = np.array([[1, 0],[0, 1/np.sqrt(2)+1j/np.sqrt(2)]])
        np.testing.assert_array_equal(t, expected_t)

    