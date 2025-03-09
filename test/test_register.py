import pytest
import numpy as np
import scipy.sparse as sps
from quantum_computing_project.register import Register
from quantum_computing_project.operations import Operations
from quantum_computing_project.constants import *

class TestRegister:

    def test_apply_gates_single(self):
        """
        Test case for applying a single gate to a one-qubit register.

        This test applies a single quantum gate (H) to a quantum register containing
        the zero state (|0⟩). It checks that after the gate is applied, the register
        is transformed into the PLUS state (|+⟩).

        Asserts:
            - The register's state after applying the gate is |1⟩.
        """
        testReg = Register(1, [ZERO])
        testGates = np.array([H])
        testReg.apply_gates(testGates)
        np.testing.assert_array_equal(testReg.reg.toarray(), PLUS.toarray())

    def test_apply_gates_multi(self):
        """
        Test case for applying multiple gates to a multi-qubit register.

        This test applies two gates (I and X) to a quantum register containing
        the states |0⟩ and |1⟩, respectively. It checks that the result of applying
        the gates results in the expected transformed states.

        Asserts:
            - The register's state after applying the gates is the tensor product
              of the |0⟩ and |0⟩ states.
        """
        testReg = Register(2, [ZERO, ONE])
        testGates = np.array([I, X])
        testReg.apply_gates(testGates)
        expected_state = Operations.sparse_tensor(ZERO, ZERO).toarray()
        np.testing.assert_array_equal(testReg.reg.toarray(), expected_state)

    def test_apply_CNOT(self):
        """
        Test case for applying a CNOT gate to a register consisting of a single qubit.

        This test applies a CNOT gate to a register whose state vector is |0⟩ and
        a control register whose state vector is |1⟩.

        Asserts:
            - The target register's state after applying the CNOT gate is |1⟩ (i.e. it's
            the same as the control register's state).
        """
        control = Register(1, [ONE])
        target = Register(1, [ZERO])
        target.apply_CNOT(control)
        np.testing.assert_array_equal(target.reg.toarray(), control.reg.toarray())


    def test_add(self):
        """
        Tests the addition of two quantum registers.

        This test checks whether adding two quantum registers correctly creates 
        a new register with the combined number of qubits and appropriate states.

        Asserts:
            - The concatenated register's quantum state matches the expected quantum state.
        """
        r1 = Register(n_qubits=2, states=[ZERO, ZERO])
        r2 = Register(n_qubits=3, states=[ZERO, ZERO, ZERO])
        result_register = r1 + r2
        expected_register = Register(5, [ZERO, ZERO, ZERO, ZERO, ZERO])
        
        np.testing.assert_array_equal(result_register._reg.toarray(), expected_register._reg.toarray())

    def test_add_err(self):
        """
        Tests the addition of a quantum register with an invalid object (str).

        Ensures that adding a Register to a non-Register object raises a TypeError.

        Asserts:
            - A TypeError is raised.
        """
        r1 = Register(n_qubits=2, states=[ZERO, ZERO])

        with pytest.raises(TypeError):
            r1 + "invalid_object"