import pytest
import numpy as np
import scipy.sparse as sps
from quantum_computing_project.register import Register

class TestRegister():

    # NON-SPARSE MATRICES TEST
    # @classmethod
    # def setup_class(cls):
    #     """Sets up zero and one state so they can be used throughout the class."""
    #     cls.zero = np.array([[1], [0]])
    #     cls.one = np.array([[0], [1]]) 

    # def test_add(self):
    #     """
    #     Tests the addition of two quantum registers.

    #     This test checks whether adding two quantum registers correctly creates 
    #     a new register with the combined number of qubits and appropriate states.

    #     Asserts:
    #         - The concatenated register's quantum state matches the expected quantum state.
    #     """
    #     r1 = Register(n_qubits=2, states=[self.zero, self.zero])
    #     r2 = Register(n_qubits=3, states=[self.zero, self.zero, self.zero])
    #     result_register = r1 + r2
    #     expected_register = Register(5, [self.zero, self.zero, self.zero, self.zero, self.zero])
        
    #     np.testing.assert_array_equal(result_register._reg, expected_register._reg)

    @classmethod
    def setup_class(cls):
        """Sets up (sparse) zero and one states so they can be used throughout the class."""
        cls.zero = sps.coo_matrix(([1], ([0], [0])), shape=(2, 1))
        cls.one = sps.coo_matrix(([1], ([1], [0])), shape=(2, 1))

    def test_add(self):
        """
        Tests the addition of two quantum registers.

        This test checks whether adding two quantum registers correctly creates 
        a new register with the combined number of qubits and appropriate states.

        Asserts:
            - The concatenated register's quantum state matches the expected quantum state.
        """
        r1 = Register(n_qubits=2, states=[self.zero, self.zero])
        r2 = Register(n_qubits=3, states=[self.zero, self.zero, self.zero])
        result_register = r1 + r2
        expected_register = Register(5, [self.zero, self.zero, self.zero, self.zero, self.zero])
        
        np.testing.assert_array_equal(result_register._reg.toarray(), expected_register._reg.toarray())

    def test_add_err(self):
        """
        Tests the addition of a quantum register with an invalid object (str).

        Ensures that adding a Register to a non-Register object raises a TypeError.

        Asserts:
            - A TypeError is raised.
        """
        r1 = Register(n_qubits=2, states=[self.zero, self.zero])

        with pytest.raises(TypeError):
            r1 + "invalid_object"