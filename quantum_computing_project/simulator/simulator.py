import numpy as np
from functools import reduce

from quantum_computing_project.register import Register
from quantum_computing_project.gate import Gate
from quantum_computing_project.operations import Operations
from quantum_computing_project.constants import *
import matplotlib.pyplot as plt
import math

class Simulator:
    """
    The Simulator class is where Deutsch-Josza, Grover's etc. will be implemented.
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
        k = math.floor((np.pi / 4) * np.sqrt(N))
        # added floor above as its mathematically correct
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
        for _ in range(t):
            grover_state.apply_gates(np.array([flip_operator]))
            grover_state.apply_gates(np.array([H]).repeat(grover_state.n_qubits))
            grover_state.apply_gates(np.array([R0]))
            grover_state.apply_gates(np.array([H]).repeat(grover_state.n_qubits))


        #plt.bar(func, grover_state.distribution())
        #plt.xlabel("State")
        #plt.ylabel("Probability")
        #plt.show()

        return grover_state.distribution()


    @staticmethod
    def grover_multiple_known_sols(func, search_object):
        """Inputs:
            func : unordered list containing all possible solutions
            search_object : list containing all known solutions"""
        N = len(func)
        n_sols = len(search_object)
        theta = np.arcsin(np.sqrt(n_sols / N))
        t = np.floor(np.pi / (4 * theta))
        #theta and t as defined by Grover's for multiple sols
        print(f"Ideal t within function: {t}")
        n_inputs = int(np.log(N) / np.log(2))
        initial_state = Register(n_inputs, [ZERO for _ in range(n_inputs)])
        initial_state.apply_gates(np.array([H]).repeat(initial_state.n_qubits))

        desired_states = []
        missing_values = []

        for i in range(len(func)):
            if func[i] in search_object:
                desired_states.append(i)

        # Check for values in search_object that are missing from func
        missing_values = [val for val in search_object if val not in func]

        if missing_values:
            raise ValueError(f"The following desired values do not exist in the data: {missing_values}")

        if not desired_states:
            raise ValueError("The desired values do not exist in the data")

        flip_data = np.ones(2 ** n_inputs)

        for state in desired_states:
            flip_data[state] = -1

        flip_operator = Gate(flip_data, np.linspace(0, (2 ** n_inputs) - 1, 2 ** n_inputs, dtype=int),np.linspace(0, (2 ** n_inputs) - 1, 2 ** n_inputs, dtype=int), True)

        R0_data = -np.ones(2 ** initial_state.n_qubits)
        R0_data[0] = 1
        R0 = Gate(R0_data,np.linspace(0, (2 ** n_inputs) - 1, 2 ** n_inputs, dtype=int),
                  np.linspace(0, (2 ** n_inputs) - 1, 2 ** n_inputs, dtype=int),
                  True)


        for _ in range(int(t)):
            initial_state.apply_gates(np.array([flip_operator]))
            initial_state.apply_gates(np.array([H]).repeat(initial_state.n_qubits))
            initial_state.apply_gates(np.array([R0]))
            initial_state.apply_gates(np.array([H]).repeat(initial_state.n_qubits))

        #print(initial_state.distribution())
        # Return the final probability distribution after Grover iterations
        return initial_state.distribution()
    @staticmethod

    def grover_mod(func, search_object, t):
        """
        This is for plotting the probability of a solution against different values of t, interesting results
        Inputs:
            func : unordered list containing all possible solutions
            search_object : list containing all known solutions
            t : number of Grover iterations to apply"""

        N = len(func)

        n_inputs = int(np.log(N) / np.log(2))
        initial_state = Register(n_inputs, [ZERO for _ in range(n_inputs)])


        initial_state.apply_gates(np.array([H]).repeat(initial_state.n_qubits))

        desired_states = []
        missing_values = []

        # Find the indices of the desired solutions in the input list
        for i in range(len(func)):
            if func[i] in search_object:
                desired_states.append(i)

        # Check for values in search_object that are missing from func
        missing_values = [val for val in search_object if val not in func]

        if missing_values:
            raise ValueError(f"The following desired values do not exist in the data: {missing_values}")

        if not desired_states:
            raise ValueError("The desired values do not exist in the data")

        # Create the Grover Oracle (marked states have a phase flip of -1)
        flip_data = np.ones(2 ** n_inputs)
        for state in desired_states:
            flip_data[state] = -1

        flip_operator = Gate(
            flip_data,
            np.linspace(0, (2 ** n_inputs) - 1, 2 ** n_inputs, dtype=int),
            np.linspace(0, (2 ** n_inputs) - 1, 2 ** n_inputs, dtype=int),
            True,
        )

        # Create the reflection about the mean operator
        R0_data = -np.ones(2 ** initial_state.n_qubits)
        R0_data[0] = 1
        R0 = Gate(
            R0_data,
            np.linspace(0, (2 ** n_inputs) - 1, 2 ** n_inputs, dtype=int),
            np.linspace(0, (2 ** n_inputs) - 1, 2 ** n_inputs, dtype=int),
            True,
        )

        # Apply Grover iterations exactly `t` times
        for _ in range(int(t)):
            initial_state.apply_gates(np.array([flip_operator]))  # Oracle
            initial_state.apply_gates(np.array([H]).repeat(initial_state.n_qubits))  # Hadamard
            initial_state.apply_gates(np.array([R0]))  # Reflection
            initial_state.apply_gates(np.array([H]).repeat(initial_state.n_qubits))  # Hadamard

        #print(initial_state.distribution())
        # Return the final probability distribution after Grover iterations
        return initial_state.distribution()

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

    def grover_any_length(func, search_object):
        """Inputs:
            func : unordered list containing all possible solutions
            search_object : list containing all known solutions
        """
        # Determine the original input length and compute the required number of qubits.
        L = len(func)
        n_inputs = int(np.ceil(np.log2(L)))
        extended_length = 2 ** n_inputs

        # If necessary, pad 'func' with dummy values (here None) so that the Hilbert space is 2^n_inputs.
        if L < extended_length:
            padded_func = func + [None] * (extended_length - L)
        else:
            padded_func = func

        n_sols = len(set(search_object))
        # Use the extended Hilbert space dimension for theta and t
        theta = np.arcsin(np.sqrt(n_sols / extended_length))
        t = np.floor(np.pi / (4 * theta))
        print(f"Ideal t within function: {t}")

        # Initialize the register in state |0...0⟩ and then apply Hadamard gates.
        initial_state = Register(n_inputs, [ZERO for _ in range(n_inputs)])
        initial_state.apply_gates(np.array([H]).repeat(initial_state.n_qubits))

        # Identify indices in the padded function that are in the search_object.
        desired_states = []
        for i in range(len(padded_func)):
            if padded_func[i] in search_object:
                desired_states.append(i)

        # Check for desired values missing from the original function (not the padded version)
        missing_values = [val for val in search_object if val not in func]
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

        # Get the full probability distribution from the extended Hilbert space.
        full_distribution = initial_state.distribution()
        # Truncate the distribution to only include the original input states.
        final_distribution = full_distribution[:L]

        # Optionally re-normalize to account for the truncation.
        norm = np.sum(final_distribution)
        if norm > 0:
            final_distribution = final_distribution / norm

        return final_distribution
    @staticmethod
    def grover_calculate(unordered_list, search_object, t):

        L = len(unordered_list)
        n_inputs = int(np.ceil(np.log2(L)))
        extended_length = 2 ** n_inputs

        # If necessary, pad list with dummy values (here None) so that the Hilbert space is 2^n_inputs.
        if L < extended_length:
            padded_func = unordered_list + [None] * (extended_length - L)
        else:
            padded_func = unordered_list

        """Inputs:
            func : unordered list containing all possible solutions
            search_object : list containing all known solutions
            t : number of Grover iterations to apply"""
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

        # Get the full probability distribution from the extended Hilbert space.
        full_distribution = initial_state.distribution()
        # Truncate the distribution to only include the original input states.
        final_distribution = full_distribution[:L]

        # Optionally re-normalize to account for the truncation.
        norm = np.sum(final_distribution)
        if norm > 0:
            final_distribution = final_distribution / norm

        return final_distribution

    @staticmethod
    def grover_initialise(unordered_list, search_vals, unique, known_n_sols):

        """ Inputs:
        unordered_list : list of unordered values to search
        search_vals : list of values to search for
         unique: boolean, if true, only unique values are searched for, if false, duplicate values are also searched for
         known_n_sols: boolean, if true, the number of solutions is known, if false, the number of solutions is unknown"""
        # Determine the original input length and compute the required number of qubits.

        if unique:
            unordered_list = set(unordered_list)

        L = len(unordered_list)
        n_inputs = int(np.ceil(np.log2(L)))
        extended_length = 2 ** n_inputs
        #
        if known_n_sols:
            n_sols = len(set(search_vals))
            theta = np.arcsin(np.sqrt(n_sols / extended_length))
            t = np.floor(np.pi / (4 * theta))
            return grover_calculate(unordered_list, search_vals, t)

        else:


        #placing constraints on the ratio of solutions to inputs
        # if n_sols/L > 0.2:
        #     raise ValueError(f"There are too many solutions to search for, try again with fewer solutions")




