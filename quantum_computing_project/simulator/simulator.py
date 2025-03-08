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
            while not found_solution:
                ts = list(range(1,T+1))
                random.shuffle(ts)

                for t in ts:
                    final_distribution = Simulator.grover_calculate(unordered_list, search_vals, t)[0]

                    threshold = 0.1 * np.max(final_distribution)
                    measured_states = np.where(final_distribution > threshold)[0]

                    # Check if the measured states are in the search values
                    matched_states = [state for state in measured_states if unordered_list[state] in search_vals]

                    # Print states matched and unmatched
                    if matched_states:
                        # return final_distribution, unordered_list
                        return final_distribution, Simulator.grover_calculate(unordered_list, search_vals, t)[1]

            T = np.ceil(T*1.2)
            if T >= max_T:
                print("No solutions found")
                return None


    @staticmethod
    def classical_search(input_list, targets):
        ind = np.where(np.isin(input_list, targets))[0]
        return ind

