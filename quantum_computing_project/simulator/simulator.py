import numpy as np
from functools import reduce

from quantum_computing_project.register import Register
from quantum_computing_project.gate import Gate
from quantum_computing_project.operations import Operations
from quantum_computing_project.constants import *
import matplotlib.pyplot as plt
import math
import random
from scipy.sparse import coo_matrix

class Simulator:
    """
    The Simulator class is where Deutsch-Jozsa, Grover's and error correction are implemented.
    """

    def __init__(self, registers):
        self.registers = registers


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
    def error_correction(error_type, num_qubit):
        """
        Simulates quantum error correction for a 3-qubit system using bit-flip or phase-flip error models.

        This method encodes a quantum state, applies a specified type of error (bit-flip or phase-flip)
        to a selected qubit, and then detects the error using syndrome measurement.

        Args:
            error_type (str): The type of error to introduce. Must be either 'bit flip' or 'phase flip'.
            num_qubit (int): The qubit index (1, 2, or 3) on which the error is applied.
                             Use 0 for no error.

        Returns:
            tuple: A syndrome measurement result as a tuple (b_syndrome, c_syndrome), indicating
                   the detected error syndrome.

        Raises:
            ValueError: If an invalid `error_type` is provided or if `num_qubit` is not in {0, 1, 2, 3}.
        """
        # Encoding
        psi = Register(3, [PLUS, ZERO, ZERO])
        psi.reg = np.dot(CNOT_12.gate, psi.reg).tocoo()
        psi.reg = np.dot(CNOT_13.gate, psi.reg).tocoo()

        # Applying errors
        if error_type == 'phase flip':
            # Apply Hadamard gate (encoding)
            psi.apply_gates(np.array([H, H, H]))
            # Apply Z error
            if num_qubit == 0:
                psi.apply_gates(np.array([I, I, I])) # (0,0) -- no error
            elif num_qubit == 1:
                psi.apply_gates(np.array([Z, I, I])) # (1,0) -- phase flip on 1st qubit
            elif num_qubit == 2:
                psi.apply_gates(np.array([I, Z, I])) # (1,1) -- phase flip on 2nd qubit
            elif num_qubit == 3:
                psi.apply_gates(np.array([I, I, Z])) # (0,1) -- phase flip on 3rd qubit
            else:
                raise ValueError("You can apply a phase flip to one of 3 qubits (1, 2 or 3) or to none (0).")
            # Apply Hadamard (decoding start)
            psi.apply_gates(np.array([H, H, H]))

        elif error_type == 'bit flip':
            # Applying X error
            if num_qubit == 0:
                psi.apply_gates(np.array([I, I, I])) # (0,0) -- no error
            elif num_qubit == 1:
                psi.apply_gates(np.array([X, I, I])) # (1,0) -- 1st qubit flipped
            elif num_qubit == 2:
                psi.apply_gates(np.array([I, X, I])) # (1,1) -- 2nd qubit flipped
            elif num_qubit == 3:
                psi.apply_gates(np.array([I, I, X])) # (0,1) -- 3rd qubit flipped
            else:
                raise ValueError("You can only flip one of 3 qubits (1, 2 or 3) or choose to flip none (0).")
        else:
            raise ValueError("The error type must be 'phase flip' or 'bit flip'.")

        # Detecting error
        reg_x = Register (2, [ZERO, ZERO])
        psi = psi + reg_x
        psi.reg = np.dot(CNOT_14.gate, psi.reg).tocoo()
        psi.reg = np.dot(CNOT_24.gate, psi.reg).tocoo()
        psi.reg = np.dot(CNOT_25.gate, psi.reg).tocoo()
        psi.reg = np.dot(CNOT_35.gate, psi.reg).tocoo()

        # Measure the syndrome
        b_syndrome = Simulator.measure_register_n(psi.reg, 4)
        c_syndrome = Simulator.measure_register_n(psi.reg, 5)
        result = (int(b_syndrome), int(c_syndrome))
        print('Syndrome found: ' + str(result))
        return result

    @staticmethod
    def measure_register_n(state, n):
        """
        Helper method for error_correction. Measures the n-th qubit in a 5-qubit state.

        The function computes the total probability for the fourth qubit being 0 or 1.
        It then returns the measurement outcome (0 or 1).
        """
        prob_0 = 0.0
        prob_1 = 0.0

        # Iterate over the nonzero elements in the state vector
        for idx, amp in zip(state.row, state.data):
            # Convert the index to a 5-bit binary string
            bits = format(idx, '05b')
            # The n-th qubit is at position n-1
            if bits[n - 1] == '0':
                prob_0 += np.abs(amp) ** 2
            else:
                prob_1 += np.abs(amp) ** 2

        # Normalise probabilities
        total = prob_0 + prob_1
        if total > 0:
            prob_0 /= total
            prob_1 /= total

        outcome = np.random.choice([0, 1], p=[prob_0, prob_1])
        return outcome




    @staticmethod
    def grover_calculate(unordered_list, search_object, t):
        """Inputs:
        unordered list : list of unordered values to search that has already been filtered by grover_initialise
        search_object : list of values to search for
        t: predetermined number of Grover iterations to apply"""

        L = len(unordered_list)
        n_inputs = int(np.ceil(np.log2(L)))
        extended_length = 2 ** n_inputs

        # If necessary, pad list with dummy values (here None) so that the Hilbert space is 2^n_inputs.
        if L < extended_length:
            padded_func = unordered_list + [None] * (extended_length - L)
        else:
            padded_func = unordered_list

        # Initialize the register in state |0...0⟩ and then apply Hadamard gates.
        initial_state = Register(n_inputs, [ZERO for _ in range(n_inputs)])
        initial_state.apply_gates(np.array([H]).repeat(initial_state.n_qubits))

        # Identify indices in the padded function that are in the search_object.
        desired_states = []
        for i in range(len(padded_func)):
            if padded_func[i] in search_object:
                desired_states.append(i)

        # Check for desired values missing from the original function (not the padded version)
        missing_values = [val for val in search_object if val not in unordered_list]
        if missing_values:
            raise ValueError(f"The following desired values do not exist in the data: {missing_values}")

        if not desired_states:
            raise ValueError("The desired values do not exist in the data")



        # Create the flip operator (oracle).
        flip_data = np.ones(extended_length)
        for state in desired_states:
            flip_data[state] = -1

        indices = np.linspace(0, extended_length - 1, extended_length, dtype=int)
        flip_operator = Gate(flip_data, indices, indices, True)

        # Create the reflection about zero operator (R0)
        R0_data = -np.ones(extended_length)
        R0_data[0] = 1
        R0 = Gate(R0_data, indices, indices, True)

        # Perform Grover iterations.
        for _ in range(int(t)):
            initial_state.apply_gates(np.array([flip_operator]))
            initial_state.apply_gates(np.array([H]).repeat(initial_state.n_qubits))
            initial_state.apply_gates(np.array([R0]))
            initial_state.apply_gates(np.array([H]).repeat(initial_state.n_qubits))



        # truncating the register to only include the original input states.
        mask = initial_state.reg.row < L
        new_data = initial_state.reg.data[mask]
        new_row = initial_state.reg.row[mask]
        new_col = initial_state.reg.col[mask]
        initial_state.reg = coo_matrix((new_data, (new_row, new_col)), shape=(L, 1))

        # Normalise the probabilities before getting a measurement and distribution
        reg_sq = initial_state.reg.power(2)
        norm = np.sqrt(reg_sq.sum())
        initial_state.reg = initial_state.reg/norm

        measurement = initial_state.measure()
        final_distribution = initial_state.distribution()

        """
        # Get the full probability distribution from the extended Hilbert space.
        full_distribution = initial_state.distribution()
        # Truncate the distribution to only include the original input states.
        final_distribution = full_distribution[:L]

        # Optionally re-normalize to account for the truncation.
        norm = np.sum(final_distribution)
        if norm > 0:
            final_distribution = final_distribution / norm
        """
        return final_distribution, measurement

    @staticmethod
    def grover_initialise(unordered_list, search_vals, known_n_sols):

        """ Inputs:
        unordered_list : list of unordered values to search
        search_vals : list of values to search for
        unique (not used): boolean, if true, only unique values are searched for, if false, duplicate values are also searched for
        known_n_sols: boolean, if true, the number of solutions is known, if false, the number of solutions is unknown"""
        # Determine the original input length and compute the required number of qubits.

        search_vals = list(set(search_vals))

        L = len(unordered_list)
        n_inputs = int(np.ceil(np.log2(L)))
        extended_length = 2 ** n_inputs
        #
        if known_n_sols:
            n_sols = len(set(search_vals))
            theta = np.arcsin(np.sqrt(n_sols / extended_length))
            t = np.floor(np.pi / (4 * theta))
            # return Simulator.grover_calculate(unordered_list, search_vals, t)[0], unordered_list
            return Simulator.grover_calculate(unordered_list, search_vals, t)
        else:
            found_solution = False
            T = 1
            max_T = np.ceil(np.sqrt(extended_length))
            while not found_solution and T <= max_T:
                t = np.random.randint(1,T+1)
                final_distribution, measured_state = Simulator.grover_calculate(unordered_list, search_vals, t)

                # Print states matched and unmatched
                if unordered_list[measured_state] in search_vals:
                    # return final_distribution, unordered_list
                    found_solution = True
                    return Simulator.grover_calculate(unordered_list, search_vals, t)

                T = np.ceil(T*1.2)
            if not found_solution:
                print("No solutions found")
                return None


    @staticmethod
    def classical_search(input_list, targets):
        ind = np.where(np.isin(input_list, targets))[0]
        return ind

