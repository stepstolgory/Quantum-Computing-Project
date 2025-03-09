import numpy as np
from functools import reduce

from quantum_computing_project.register import Register
from quantum_computing_project.gate import Gate
from quantum_computing_project.operations import Operations
from quantum_computing_project.constants import *

class Simulator:
    """
    The Simulator class is where Deutsch-Jozsa, Grover's etc. will be implemented.
    """

    def __init__(self, registers):
        self.registers = registers

    @staticmethod
    def grover_simplified(func, search_object):

        """
        - The number of inputs (qubits) into the register is ln(N)/ln(2)
        - Number of iterations is (pi/4)*sqrt(N)
        - Performs the initial step of Grover's algorithm, applying Hadamard Gate n times to n qubits to create a superposition state
        """

        N = len(func)
        n_inputs = int(np.log(N) / np.log(2))
        k = (np.pi / 4) * np.sqrt(N)
        print("Initialising first step of Grover!!!")
        grover_state = Register(n_inputs, [ZERO for _ in range(n_inputs)])
        grover_state.apply_gates(np.array([H]).repeat(grover_state.n_qubits))

        # TODO: Need to add oracle step
        # ***Can't just do I - 2|s⟩⟨s| since that is not reflective of quantum computing processes, must apply gates to achieve the same effect
        # Identify target state from all other states superimposed within the register
        # Flip all 0s in the target state by applying X gates to them (not sure how)
        # Apply Hadamard on last qubit of (altered) target state
        # Apply multi-controlled X on state, control bits are all the other bits except for the final one
        # Flip the originally flipped 0s back by applying X gates to the same set of qubits

        # loop is currently for searching only one object to save time
        # TODO: Add multiple search functionality
        desired_state = None
        for i in range(0, len(func)):
            if func[i] == search_object:
                desired_state = i
                break

        if not desired_state:
            raise ValueError("The desired value does not exist in the data")

        # Generate a matrix (gate) which will flip the amplitude of the desired state
        flip_data = np.ones(2 ** n_inputs)
        flip_data[desired_state] = -1
        flip_operator = Gate(flip_data, np.linspace(0, (2 ** n_inputs) - 1, 2 ** n_inputs, dtype=int),
                             np.linspace(0, (2 ** n_inputs) - 1, 2 ** n_inputs, dtype=int), True)

        # the R0 gate is a negative 2^n identity matrix with the top left element as positive 1
        R0_data = -np.ones(2 ** grover_state.n_qubits)
        R0_data[0] = 1
        R0 = Gate(R0_data, np.linspace(0, (2 ** n_inputs) - 1, 2 ** n_inputs, dtype=int),
                  np.linspace(0, (2 ** n_inputs) - 1, 2 ** n_inputs, dtype=int), True)

        # TODO: Need to implement R0 as actual quantum gates, not a custom made matrix
        # Diffuser step, the inversion operator R0 has to be defined as a separate gate with dimensions depending on the register size
        # Apply gates as follows: Hadamard (n times) --> R0 --> Hadamard (n times)
        # ***Can't just do 2|u⟩⟨u| - I for R0 since that is not reflective of quantum computing processes, must apply gates to achieve the same effect

        e = 0
        while e <= k:
            grover_state.apply_gates(np.array([flip_operator]))
            grover_state.apply_gates(np.array([H]).repeat(grover_state.n_qubits))
            grover_state.apply_gates(np.array([R0]))
            grover_state.apply_gates(np.array([H]).repeat(grover_state.n_qubits))
            e += 1

        # plt.plots(func, grover_state.distribution())

        return grover_state.distribution()

    @staticmethod
    def deutsch_jozsa(func, n_inputs):
        """Performs the Deutsch-Jozsa Algorithm for a given function with n inputs.

        Args:
            func (np.array): An array of outputs of the given function in lexicographical order. The function must be either balanced or constant.
            n_inputs (int): Number of inputs the function takes. (2^n_inputs outputs)

        Returns:
            bool: Return True if the function is balanced and False otherwise
        """
        # print("Initialising the Deutsch-Jozsa Algorithm!!!")

        # Initialises the required registers
        reg_x = Register(n_inputs, [ZERO for _ in range(n_inputs)])
        reg_y = Register(1, [ONE])
        psi = reg_x + reg_y
        reg_x.apply_gates(np.array([H]).repeat(reg_x.n_qubits))
        reg_y.apply_gates(np.array([H]))

        results = []

        # Performs an XOR equivalent between each output of the function and the y register
        for val in func:
            results.append(reg_y.apply_CNOT_dj(val))

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

    @staticmethod
    def nine_qubit_shor(phase_flip, bit_flip, bit_blocks=None, flipped_bit=None):
        """
        The nine qubit Shor code.

        Args:
            - phase_flip (bool): If True, phase-flip error is applied.
            - bit_flip (bool): If True, bit-flip error is applied.
            - bit_blocks (list): List of blocks (1, 2 or 3) where a bit-flip error should occur
                                 (at most one bit-flip error per block).
            - flipped_bit (list): Which bit should be flipped in each block (1, 2 or 3).
        """
        if (phase_flip == False) and (bit_flip == False):
            # Prepare start state: A|0⟩+B|1⟩
            psi = (Register(1, [ZERO]))
            psi.apply_gates(np.array([H]))

            # Outer encoding (phase flip)
            reg_1a = Register(1, [ZERO])
            reg_1b = Register(1, [ZERO])
            reg_1a.apply_CNOT(psi)
            reg_1b.apply_CNOT(psi)
            psi.apply_gates(np.array([H]))
            reg_1a.apply_gates(np.array([H]))
            reg_1b.apply_gates(np.array([H]))

            # Inner encoding (bit flip)
            reg_2a = Register(1, [ZERO])
            reg_2b = Register(1, [ZERO])
            reg_2a.apply_CNOT(psi)
            reg_2b.apply_CNOT(psi)

            reg_2c = Register(1, [ZERO])
            reg_2d = Register(1, [ZERO])
            reg_2c.apply_CNOT(reg_1a)
            reg_2d.apply_CNOT(reg_1a)

            reg_2e = Register(1, [ZERO])
            reg_2f = Register(1, [ZERO])
            reg_2e.apply_CNOT(reg_1b)
            reg_2f.apply_CNOT(reg_1b)

            # Detecting phase flip
            reg_3a = Register(1, [PLUS])
            reg_3b = Register(1, [PLUS])
            la = [psi, reg_2a, reg_2b, reg_1a, reg_2c, reg_2d]
            lb = [reg_1a, reg_2c, reg_2d, reg_1b, reg_2e, reg_2f]
            for reg in la:
                reg.apply_CNOT(reg_3a)
            for reg in lb:
                reg.apply_CNOT(reg_3b)
            reg_3a.apply_gates(np.array([H]))
            reg_3b.apply_gates(np.array([H]))
            phase_result = (int(reg_3b.measure()), int(reg_3a.measure()))

            # Detecting bit flip
            # Block 1
            reg_4a = Register(1, [ZERO])
            reg_4b = Register(1, [ZERO])
            reg_4a.apply_CNOT(psi)
            reg_4a.apply_CNOT(reg_2a)
            reg_4b.apply_CNOT(reg_2a)
            reg_4b.apply_CNOT(reg_2b)
            block1_bit_result = (int(reg_4b.measure()), int(reg_4a.measure()))
            # Block 2
            reg_4c = Register(1, [ZERO])
            reg_4d = Register(1, [ZERO])
            reg_4c.apply_CNOT(reg_1a)
            reg_4c.apply_CNOT(reg_2c)
            reg_4d.apply_CNOT(reg_2c)
            reg_4d.apply_CNOT(reg_2d)
            block2_bit_result = (int(reg_4d.measure()), int(reg_4c.measure()))
            # Block 3
            reg_4e = Register(1, [ZERO])
            reg_4f = Register(1, [ZERO])
            reg_4e.apply_CNOT(reg_1b)
            reg_4e.apply_CNOT(reg_2e)
            reg_4f.apply_CNOT(reg_2e)
            reg_4f.apply_CNOT(reg_2f)
            block3_bit_result = (int(reg_4f.measure()), int(reg_4e.measure()))
            print('Phase-flip error syndrome: ' + str(phase_result))
            print('Block 1 bit-flip error syndrome: ' + str(block1_bit_result))
            print('Block 2 bit-flip error syndrome: ' + str(block2_bit_result))
            print('Block 3 bit-flip error syndrome: ' + str(block3_bit_result))

        elif (phase_flip == False) and (bit_flip == True):

            if (bit_blocks is None) and (flipped_bit is None):
                # Prepare start state: A|0⟩+B|1⟩
                psi = (Register(1, [ZERO]))
                psi.apply_gates(np.array([H]))

                # Outer encoding (phase flip)
                reg_1a = Register(1, [ZERO])
                reg_1b = Register(1, [ZERO])
                reg_1a.apply_CNOT(psi)
                reg_1b.apply_CNOT(psi)
                psi.apply_gates(np.array([H]))
                reg_1a.apply_gates(np.array([H]))
                reg_1b.apply_gates(np.array([H]))

                # Inner encoding (bit flip)
                reg_2a = Register(1, [ZERO])
                reg_2b = Register(1, [ZERO])
                reg_2a.apply_CNOT(psi)
                reg_2b.apply_CNOT(psi)

                reg_2c = Register(1, [ZERO])
                reg_2d = Register(1, [ZERO])
                reg_2c.apply_CNOT(reg_1a)
                reg_2d.apply_CNOT(reg_1a)
                reg_2d.apply_gates(np.array([X]))

                reg_2e = Register(1, [ZERO])
                reg_2f = Register(1, [ZERO])
                reg_2e.apply_CNOT(reg_1b)
                reg_2f.apply_CNOT(reg_1b)

                # Detecting phase flip
                reg_3a = Register(1, [PLUS])
                reg_3b = Register(1, [PLUS])
                la = [psi, reg_2a, reg_2b, reg_1a, reg_2c, reg_2d]
                lb = [reg_1a, reg_2c, reg_2d, reg_1b, reg_2e, reg_2f]
                for reg in la:
                    reg.apply_CNOT(reg_3a)
                for reg in lb:
                    reg.apply_CNOT(reg_3b)
                reg_3a.apply_gates(np.array([H]))
                reg_3b.apply_gates(np.array([H]))
                phase_result = (int(reg_3b.measure()), int(reg_3a.measure()))

                # Detecting bit flip
                # Block 1
                reg_4a = Register(1, [ZERO])
                reg_4b = Register(1, [ZERO])
                reg_4a.apply_CNOT(psi)
                reg_4a.apply_CNOT(reg_2a)
                reg_4b.apply_CNOT(reg_2a)
                reg_4b.apply_CNOT(reg_2b)
                block1_bit_result = (int(reg_4b.measure()), int(reg_4a.measure()))
                # Block 2
                reg_4c = Register(1, [ZERO])
                reg_4d = Register(1, [ZERO])
                reg_4c.apply_CNOT(reg_1a)
                reg_4c.apply_CNOT(reg_2c)
                reg_4d.apply_CNOT(reg_2c)
                reg_4d.apply_CNOT(reg_2d)
                block2_bit_result = (int(reg_4d.measure()), int(reg_4c.measure()))
                # Block 3
                reg_4e = Register(1, [ZERO])
                reg_4f = Register(1, [ZERO])
                reg_4e.apply_CNOT(reg_1b)
                reg_4e.apply_CNOT(reg_2e)
                reg_4f.apply_CNOT(reg_2e)
                reg_4f.apply_CNOT(reg_2f)
                block3_bit_result = (int(reg_4f.measure()), int(reg_4e.measure()))
                print('Phase-flip error syndrome: ' + str(phase_result))
                print('Block 1 bit-flip error syndrome: ' + str(block1_bit_result))
                print('Block 2 bit-flip error syndrome: ' + str(block2_bit_result))
                print('Block 3 bit-flip error syndrome: ' + str(block3_bit_result))

            elif bit_blocks is None:
                ...