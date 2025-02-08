from quantum_computing_project.simulator import Simulator
from quantum_computing_project.register import Register
from quantum_computing_project.operations import Operations
from quantum_computing_project.constants import *
import numpy as np
# TODO: add tests to check errors are correctly raised

class TestSimulator:
    """Test suite for the Simulator class."""

    def test_apply_gates_single(self):
        """
        Test case for applying a single gate to a one-qubit register.

        This test applies a single quantum gate (X) to a quantum register containing
        the zero state (|0⟩). It checks that after the gate is applied, the register
        is transformed into the one state (|1⟩).

        Asserts:
            - The register's state after applying the gate is |1⟩.
        """
        testReg = Register(1, [ZERO])
        testGates = np.array([X])
        testSim = Simulator(testReg)
        testSim.apply_gates(testGates, testReg)
        np.testing.assert_array_equal(testReg.reg.toarray(), ONE.toarray())

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
        testSim = Simulator(testReg)
        testSim.apply_gates(testGates, testReg)
        expected_state = Operations.sparse_tensor(ZERO, ZERO).toarray()
        np.testing.assert_array_equal(testReg.reg.toarray(), expected_state)