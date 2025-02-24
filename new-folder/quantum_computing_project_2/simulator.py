import numpy as np
from functools import reduce
import sys
sys.path.append("C:/Users/Sean/Desktop/QCP")
from quantum_computing_project.register import Register
from quantum_computing_project.gate import Gate
from quantum_computing_project.operations import Operations
from quantum_computing_project.constants import *

class Simulator:
    """
    The Simulator class is where Deutsch-Josza, Grover's etc. will be implemented.
    """

    def __init__(self, registers):
        self.registers = registers

    @staticmethod
    def deutsch_josza(func, n_inputs):
        """Performs the Deutsch-Josza Algorithm for a given function with n inputs.

        Args:
            func (np.array): An array of outputs of the given function in lexicographical order. The function must be either balanced or constant.
            n_inputs (int): Number of inputs the function takes. (2^n_inputs outputs)

        Returns:
            bool: Return True if the function is balanced and False otherwise
        """
        print("Initialising the Deutsch-Josza Algorithm!!!")

        # Initialises the required registers
        reg_x = Register(n_inputs, [ZERO for _ in range(n_inputs)])
        reg_y = Register(1, [ONE])
        psi = reg_x + reg_y
        reg_x.apply_gates(np.array([H]).repeat(reg_x.n_qubits))
        reg_y.apply_gates(np.array([H]))

        results = []

        # Performs an XOR equivalent between each output of the function and the y register
        for val in func:
            results.append(reg_y.apply_CNOT(val))

        # Multiplies each state of the x register by its corresponding result for f(x) XOR reg_y
        results = np.array(results)
        final_res = (reg_x.reg.toarray() * results.reshape(results.size // 2, 2)).reshape(
            results.size, 1
        )
        psi.reg = sps.coo_matrix(final_res)

        # Removes the superposition
        psi.apply_gates(np.array([H]).repeat(psi.n_qubits))

        # Returns all states whose amplitudes are non-zero
        states = [format(index, "b") for index in np.where(~np.isclose(psi.reg.toarray(), 0))[0]]

        # If any of the qubits in register x are non-zero then the function is balanced
        balanced = any("1" in state[:-1] for state in states)

        return balanced
