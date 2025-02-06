# First version of quantum computer simulation

"""This version of the simulation steps away from the single qubit implementation and acts everything upon registers."""

import numpy as np
import scipy.linalg as spl
import scipy.sparse as sps
import time
from functools import reduce

rng = np.random.default_rng()


class Simulation:
    registers = []

    def __init__(self):
        print("Simulation Initialised")

    def tensor(self, A, B):
        """
        Computes the tensor product between two matrices
        A (X) B = each element of Aij*B so m x n (X) p x q gives mp x nq matrix

        Args:
            A (numpy.array): 2D numpy array which represents the matrix on the right of the tensor product
            B (_type_): 2D numpy array which represents the matrix on the left of the tensor product

        Returns:
            np.array: 2D numpy array which represents the matrix resulting from the tensor product A(X)B
        """
        m, n = A.shape
        p, q = B.shape

        A_flat = A.flatten()
        B_flat = B.flatten()

        outer = A_flat[:, None] * B_flat[None, :]
        outer4d = outer.reshape(m, n, p, q)
        outermpnq = outer4d.transpose(0, 2, 1, 3)
        result = outermpnq.reshape(m * p, n * q)

        return result

    def sparse_tensor(self, A, B):
        """
        Same as above except the inputs and outputs are sparse COO matrices. May or may not be useful
        """
        m, n = A.shape
        p, q = B.shape

        new_row = []
        new_col = []
        new_data = []

        for ai, aj, aval in zip(A.row, A.col, A.data):
            for bi, bj, bval in zip(B.row, B.col, B.data):
                row_id = ai * p + bi
                col_id = aj * q + bj
                val = aval * bval

                new_row.append(row_id)
                new_col.append(col_id)
                new_data.append(val)

        C = sps.coo_matrix((new_data, (new_row, new_col)), shape=(m * p, n * q))

        return C

    def power_tensor(self, A, p):
        """
        Performs the tensor product p times between the same matrix
        A ^ (X)p = A (X) A (X) A (x) A ... (X) A for p matrices

        Args:
            A (numpy.array): 2D numpy array which represents the input matrix
            p (integer): Number of matrices between which the power tensor is perfomed (p-1 operations)

        Returns:
            numpy.array: 2D array which represents the final result of the operations
        """
        inter_mat = A
        start_time = time.time()
        for _ in range(p - 1):
            inter_mat = self.tensor(A, inter_mat)
        end_time = time.time()
        # print(f"Time taken: {end_time-start_time}.")
        return inter_mat

    def sparse_power_tensor(self, A, p):
        # TODO: Sparse matrices not implemented yet
        inter_mat = A
        start_time = time.time()
        for _ in range(p):
            inter_mat = self.sparse_tensor(A, inter_mat)
        end_time = time.time()
        print(f"Time taken: {end_time-start_time}.")
        return inter_mat

    def apply_gates(self, gates, register):
        # Need to make sure that the number of gates matches the number of qubits in the register
        # Gates should be an array of size n and contain the gate objects
        # Qubits_pos should also be an array of the same size n which has the position of the qubit in the register on which the gate acts.
        # so if you want to act on the second qubit with a hadamard gate. The gates array will be [H (object)], qubits_pos will have [1]
        # Reg will be the register on which the gates are applied
        # A tensor product needs to be done between all of the gates and it needs to slot in an I gate if no gate is specified for that position

        if isinstance(gates, Gate):

            resulting_gate = self.power_tensor(gates.gate, register.n_qubits)
            register.reg = np.dot(resulting_gate, register.reg)

        elif isinstance(gates, np.ndarray) and all(isinstance(g, Gate) for g in gates):

            if gates.size != register.n_qubits:
                raise ValueError(
                    "The number of gates must match the number of qubits in the register!!!"
                )

            gates = np.array([gate.gate for gate in gates])

            resulting_gate = reduce(self.tensor, gates)

            register.reg = np.dot(resulting_gate, register.reg)

        else:
            raise TypeError(
                "The 'gates' parameter must either be a single instance of the Gate class, or a numpy.ndarray of instances of Gate."
            )

    def apply_CNOT(self, CNOT, control_qubit, register):
        """Applies a CNOT gate to a register consisting of a single qubit

        Args:
            CNOT (Gate): An instance of the Gate class which represents the 2 qubit CNOT gate
            control_qubit (numpy.array): State of the qubit that controls the gate. (zero or one)
            register (Register): An instance of the Register class which represents a register with a single qubit that will be acted on by the CNOT gate

        Returns:
            numpy.array: The array of the separate amplitudes of the zero state and the one state of the second (original) qubit.
        """
        reg_with_control = self.tensor(control_qubit, register)
        final_reg = np.dot(CNOT.gate, reg_with_control)

        # Factor out the second qubit
        zero_amp = np.sum(final_reg[0::2])
        one_amp = np.sum(final_reg[1::2])

        # var.qubit = np.array([[zero_amp], [one_amp]])
        resulting_amps = np.array([[zero_amp], [one_amp]])

        return resulting_amps

    def deutsch_josza(self, func, n_inputs):
        """Performs the Deutsch-Josza Algorithm for a given funtion with n inputs.

        Args:
            func (np.array): An array of outputs of the given function in lexicographical order. The function must be either balanced or constant.
            n_inputs (int): Number of inputs the function takes. (2^n_inputs outputs)

        Returns:
            bool: Return True if the function is balanced and False otherwise
        """
        print("Initialising the Deutsch-Josza Algorithm!!!")

        reg_x = Register(n_inputs, [zero for _ in range(n_inputs)])
        reg_y = Register(1, [one])
        psi = reg_x + reg_y

        simulation.apply_gates(H, psi)
        simulation.apply_gates(H, reg_x)
        simulation.apply_gates(H, reg_y)

        results = []
        for val in func:
            results.append(simulation.apply_CNOT(CNOT_2, val, reg_y.reg))

        results = np.array(results)

        psi.reg = (reg_x.reg * results.reshape(results.size // 2, 2)).reshape(
            results.size, 1
        )
        simulation.apply_gates(H, psi)

        states = [format(index, "b") for index in np.where(~np.isclose(psi.reg, 0))[0]]

        balanced = any("1" in state[:-1] for state in states)

        return balanced


class Register(Simulation):
    """
    Contains any number of qubits in a register together
    """

    reg_id = 0

    def __init__(self, n_qubits, states):
        """
        Initialises a register

        Args:
            n_qubits (int): The number of qubits which are in a register together
        """
        self.reg_id = Register.reg_id
        Register.reg_id += 1

        self.n_qubits = n_qubits

        self._reg = None if self.n_qubits == 0 else reduce(self.tensor, states)

        Simulation.registers.append(self)

    @property
    def reg(self):
        return self._reg

    @reg.setter
    def reg(self, value):
        self._reg = value

    def partial_measure(self, pos):
        pass

    @property
    def dirac_representation(self):
        dr = f"{np.around(self.reg[0, 0], decimals=6)}|0>"
        for i in range(1, 2**self.n_qubits):
            dr += f" + {np.around(self.reg[i, 0], decimals = 6)}|{i}>"
        return dr

    def __str__(self):
        try:
            self.dirac_representation
            return f"Register {self.reg_id} contains {self.n_qubits} qubits with representation of\n {self.dirac_representation}"
        except TypeError:
            return f"Register {self.reg_id} contains no qubits!!!"

    def __add__(self, other):
        if isinstance(other, Register):
            new_n_bits = self.n_qubits + other.n_qubits
            newReg = Register(new_n_bits, [zero for _ in range(new_n_bits)])
            newReg.reg = self.tensor(self.reg, other.reg)
            return newReg
        else:
            raise TypeError("The things added must be instances of the same class.")


class Gate(Simulation):

    def __init__(self, data, row, column, sparse):
        """Initialises a quantum gate

        Args:
            data (numpy.array): Array containing non-zero data values in the matrix
            row (numpy.array): Array containing row indeces where the values are placed, in order of the values
            column (numpy.array): Array containing column indeces where the values are places. Together with the row array has coordinates for the data points.
            sparse (bool): Signals whether the matrix should be treated as sparse
        """
        self.sparse = sparse
        if self.sparse:
            # TODO: Add functionality to create sparse COO matrix
            raise NotImplementedError("Sparse matrices are currently not implemented.")
        else:
            self.gate = np.zeros((row.max() + 1, column.max() + 1))
            for i in range(data.size):
                self.gate[row[i], column[i]] = data[i]


if __name__ == "__main__":

    simulation = Simulation()

    data_i = np.array([1, 1])
    data_y = np.array([-1j, 1j])
    data_z = np.array([1, -1])
    data_s = np.array([1, 1j])
    data_t = np.array([1, np.exp(np.pi / 4 * 1j)])
    data_h = 1 / np.sqrt(2) * np.array([1, 1, 1, -1])
    data_cnot2 = np.array([1, 1, 1, 1])

    row = np.array([0, 1])
    row_h = np.array([0, 0, 1, 1])
    col_h = np.array([0, 1, 0, 1])
    main_diag_col = np.array([0, 1])
    off_diag_col = np.array([1, 0])
    row_cnot2 = np.array([0, 1, 2, 3])
    col_cnot2 = np.array([0, 1, 3, 2])

    # FIXME: Gates with complex components are commented out for now as I need to deal with the complex casting

    I = Gate(data_i, row, main_diag_col, False)
    X = Gate(data_i, row, off_diag_col, False)
    # Y = Gate(data_y, row, off_diag_col, False)
    Z = Gate(data_z, row, main_diag_col, False)
    # S = Gate(data_s, row, main_diag_col, False)
    # T = Gate(data_t, row, main_diag_col, False)
    H = Gate(data_h, row_h, col_h, False)
    CNOT_2 = Gate(data_cnot2, row_cnot2, col_cnot2, False)

    zero = np.array([[1], [0]])
    one = np.array([[0], [1]])
# region Test Functions for Deutsch-Jocza
# f_constant_zero = [zero, zero, zero, zero, zero, zero, zero, zero]
# f_constant_one = [one, one, one, one, one, one, one, one]

# f_balanced1 = [zero, zero, one, one]
# f_balanced2 = [zero, zero, one, one, zero, zero, one, one]
# f_balanced3 = [zero, zero, one, one, zero, zero, one, one]
# f_balanced4 = [zero, one, one, zero, one, zero, zero, one]

# f_constant_zero_4 = [
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
# ]
# f_constant_one_4 = [
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
# ]

# f_balanced_4 = [
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
# ]

# f_balanced_5 = [
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
# ]

# f_constant_zero_5 = [
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
# ]
# f_constant_one_5 = [
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
# ]

# f_balanced_large = [
#     # 128 zeros (16 lines of 8 zeros each)
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     zero,
#     # 128 ones (16 lines of 8 ones each)
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
#     one,
# ]

# f_big_as_fuck = [zero] * 2**12 + [one] * 2**12

# if simulation.deutsch_josza(f_balanced1, 2):
#     print("F1 The function is balanced.")
# else:
#     print("F1 The function is constant.")

# if simulation.deutsch_josza(f_balanced2, 3):
#     print("F2 The function is balanced.")
# else:
#     print("F2 The function is constant.")

# if simulation.deutsch_josza(f_constant_zero, 3):
#     print("F3 The function is balanced.")
# else:
#     print("F3 The function is constant.")

# if simulation.deutsch_josza(f_balanced3, 3):
#     print("F4 The function is balanced.")
# else:
#     print("F4 The function is constant.")

# if simulation.deutsch_josza(f_balanced4, 3):
#     print("F5 The function is balanced.")
# else:
#     print("F5 The function is constant.")

# if simulation.deutsch_josza(f_constant_one, 3):
#     print("F6 The function is balanced.")
# else:
#     print("F6 The function is constant.")

# if simulation.deutsch_josza(f_constant_zero_4, 4):
#     print("F7 The function is balanced.")
# else:
#     print("F7 The function is constant.")

# if simulation.deutsch_josza(f_constant_one_4, 4):
#     print("F8 The function is balanced.")
# else:
#     print("F8 The function is constant.")

# if simulation.deutsch_josza(f_balanced_4, 4):
#     print("F9 The function is balanced.")
# else:
#     print("F9 The function is constant.")

# if simulation.deutsch_josza(f_constant_zero_5, 5):
#     print("F10 The function is balanced.")
# else:
#     print("F10 The function is constant.")

# if simulation.deutsch_josza(f_constant_one_5, 5):
#     print("F11 The function is balanced.")
# else:
#     print("F11 The function is constant.")

# if simulation.deutsch_josza(f_balanced_5, 5):
#     print("F12 The function is balanced.")
# else:
#     print("F12 The function is constant.")

# if simulation.deutsch_josza(f_balanced_large, 8):
#     print("F13 The function is balanced.")
# else:
#     print("F13 The function is constant.")

# if simulation.deutsch_josza(f_big_as_fuck, 13):
#     print("F14 The function is balanced.")
# else:
#     print("F14 The function is constant.")
# endregion
