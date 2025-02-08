import numpy as np
from functools import reduce
import scipy.sparse as sps

from quantum_computing_project.register import Register
from quantum_computing_project.gate import Gate
from quantum_computing_project.operations import Operations

class Simulator:

    def __init__(self, registers):
        self.registers = registers

    @staticmethod
    def apply_gates(gates, register):
        """Applies n gates on a register with n qubits.

        Args:
            gates (numpy.array of Gate objects): The gates that are applied to the register. The array must
                                                 be of the same length as the number of qubits in the register.
            register (Register): Register which the gates are applied to.

        Raises:
            ValueError: Makes sure the number of gates matches the number of qubits in the register.
            TypeError: Makes sure that the `gates` variable is of the correct type.
        """
        if gates.size != register.n_qubits:
             raise ValueError(
                 "The number of gates must match the number of qubits in the register!!!"
             )

        gates = np.array([g.gate for g in gates])
        resulting_gate = reduce(Operations.sparse_tensor, gates)
        register.reg = np.dot(resulting_gate, register.reg)

        # if isinstance(gates, Gate):
        #
        #     resulting_gate = Operations.power_tensor(gates.gate, register.n_qubits)
        #     register.reg = np.dot(resulting_gate, register.reg)
        #
        # elif isinstance(gates, np.ndarray) and all(isinstance(g, Gate) for g in gates):
        #
        #     if gates.size != register.n_qubits:
        #         raise ValueError(
        #             "The number of gates must match the number of qubits in the register!!!"
        #         )
        #
        #     gates = np.array([gate.gate for gate in gates])
        #
        #     resulting_gate = reduce(Operations.tensor, gates)
        #
        #     register.reg = np.dot(resulting_gate, register.reg)
        #
        # else:
        #     raise TypeError(
        #         "The 'gates' parameter must either be a single instance of the Gate class, or a numpy.ndarray of instances of Gate."
        #     )
