# First version of quantum computer simulation
# TODO: Add qubit class
# TODO: Add gates class
# TODO: Add register class
# TODO: Add simulation class

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
        A (X) B = each element of Aij*B so n x n (X) n x n gives n^2 x n^2 matrix

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
            p (integer): Number of times that the tensor product is performed (p+1 matrices)

        Returns:
            numpy.array: 2D array which represents the final result of the operations
        """
        inter_mat = A
        start_time = time.time()
        for _ in range(p):
            inter_mat = self.tensor(A, inter_mat)
        end_time = time.time()
        print(f"Time taken: {end_time-start_time}.")
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

    def apply_gates(self, gates, qubits_pos, register):
        # Need to make sure that the number of gates matches the number of qubits in the register
        # Gates should be an array of size n and contain the gate objects
        # Qubits_pos should also be an array of the same size n which has the position of the qubit in the register on which the gate acts.
        # so if you want to act on the second qubit with a hadamard gate. The gates array will be [H (object)], qubits_pos will have [1]
        # Reg will be the register on which the gates are applied
        # A tensor product needs to be done between all of the gates and it needs to slot in an I gate if no gate is specified for that position

        # check gates and qubits_pos have the same shape
        if gates.shape != qubits_pos.shape:
            raise ValueError(
                "The shape of the gates array must match the shape of the qubits_pos array!!!"
            )
        for i, qubit in enumerate(register.qubits[qubits_pos]):

            # if gates[i].gate.shape[1] != qubit.amplitudes.shape[0]:
            #     pad_width = abs(gates[i].gate.shape[1] - qubit.amplitudes.shape[0])
            #     padded_amps = np.pad(
            #         qubit.amplitudes, pad_width, mode="constant", constant_values=0
            #     )
            qubit.qubit = np.dot(gates[i].gate, qubit.qubit)

    def apply_CNOT(self, CNOT, control, var, full=True):
        # TODO: Finish the implementation of multi-qubit gates
        mini_reg = self.tensor(control.qubit, var.qubit)
        res = np.dot(CNOT.gate, mini_reg)
        if full:
            return res
        else:
            return res


class Qubit:

    def __init__(self, amplitudes):
        """Initialises single qubits

        Args:
            amplitudes (numpy.array): Contains the amplitudes for the (|0>, |1>)
            superposition state, either could be 0 but the total magnitude must add to 1
        """
        self.basis_states = np.array([[1], [0]]), np.array([[0], [1]])

        self._qubit = np.dot(amplitudes, self.basis_states)

        self.reg = None
        self.pos = None

        self._entangled = False
        self._pair = None

        # Ensures that the total probability of the qubit remains at 1 (equivalent to taking out a global phase factor)
        # FIXME: I am not 100% sure that it's equivalent so please correct this if I'm wrong.
        self.normalise()

    @property
    def qubit(self):
        # Returns the superposed qubit
        return self._qubit

    @qubit.setter
    def qubit(self, value):
        self._qubit = value

    @property
    def prob(self):
        # Probabilities of collapsing to each state
        if not self.entangled:
            return self._qubit**2
        else:
            raise NotImplementedError(
                "The qubit is entangled, not sure how to deal with the probabilities yet"
            )

    # I am not 100% sure whether we need a specific check for whether a qubit is entangled, but here it is. Along with its pair and a way to set the two values
    @property
    def entangled(self):
        # Returns whether the qubit is entangled
        return self._entangled

    @entangled.setter
    def entangled(self, value):
        """Entangles the qubit with another one and indicates what the pair is

        Args:
            value (numpy.array): Contains the information on which qubit this one is entangled with. [register, position]
        """
        self._entangled = True
        self._pair = value

    @property
    def pair(self):
        # Returns the qubit that this one is entangled with
        return self._pair

    def measure(self):
        # Collapses into one of the two states
        if not self._entangled:
            return (
                self.basis_states[0]
                if rng.random() <= self.prob[0]
                else self.basis_states[1]
            )

    def normalise(self):
        # Normalises the wavefunction

        mag = spl.norm(self._qubit)
        if not np.isclose(mag, 1.0):
            self._qubit /= mag

    def __str__(self):
        return f"Qubit at position {self.pos} in register {self.reg} with representation of\n {self.amplitudes[0]}|0> + {self.amplitudes[1]}|1>"


class Register(Simulation):
    """
    Contains any number of qubits in a register together
    """

    reg_id = 0

    def __init__(self, qubits):
        """
        Initialises a register

        Args:
            qubits (numpy.array): Holds values of type object. The qubits which are in a register together
        """
        self.reg_id = Register.reg_id
        Register.reg_id += 1

        self.qubits = qubits
        self.n_qubits = qubits.size

        for i in range(self.n_qubits):
            self.qubits[i].reg = self.reg_id
            self.qubits[i].pos = i

        Simulation.registers.append(self)

    @property
    def reg(self):
        """
        Calculates the matrix for the entire register

        Returns:
            numpy.array: Column matrix of size 2^self.n_qubits
        """
        if self.n_qubits == 0:
            return None

        qubit_states = [qubit.qubit for qubit in self.qubits]

        return reduce(self.tensor, qubit_states)

    def add_qubit(self, qubit):
        """Appends a qubit to the register

        Args:
            qubit (object): Instance of the Qubit class which you want to add

        Returns:
            numpy.array: The complete matrix of the new register
        """
        print("ADDING QUBITS ARE NOT FULLY IMPLEMENTED YET!!")
        self.qubits = np.append(self.qubits, qubit)
        self.n_qubits += 1
        return self.reg

    def remove_qubit(self, pos):
        """Removes a qubits from the register

        Args:
            pos (int): Index of the qubit you want removed

        Returns:
              numpy.array: The complete matrix of the new register
        """
        print("REMOVING QUBITS ARE NOT FULLY IMPLEMENTED YET!!")
        self.qubits = np.delete(self.qubits, pos)
        self.n_qubits -= 1
        return self.reg

    def __str__(self):
        try:
            dirac_representation = f"{np.around(self.reg[0, 0], decimals=6)}|0>"
            for i in range(1, 2**self.n_qubits):
                dirac_representation += (
                    f" + {np.around(self.reg[i, 0], decimals = 6)}|{i}>"
                )
            return f"Register {self.reg_id} contains {self.n_qubits} qubits with representation of\n {dirac_representation}"
        except TypeError:
            return f"Register {self.reg_id} contains no qubits!!!"


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
    # TODO: Will need a CNOT gate, try find a way to make this automatic and not harcoding it in

    I = Gate(data_i, row, main_diag_col, False)
    X = Gate(data_i, row, off_diag_col, False)
    # Y = Gate(data_y, row, off_diag_col, False)
    Z = Gate(data_z, row, main_diag_col, False)
    # S = Gate(data_s, row, main_diag_col, False)
    # T = Gate(data_t, row, main_diag_col, False)
    H = Gate(data_h, row_h, col_h, False)
    CNOT_2 = Gate(data_cnot2, row_cnot2, col_cnot2, False)
    print(H.gate)

    zero = np.array([[1], [0]])
    one = np.array([[0], [1]])

    print("Testing the Deutsch-Josza Algorithm!!!")

    # Create |001> register
    qubit0 = Qubit(np.array([1, 0]))
    qubit1 = Qubit(np.array([1, 0]))
    qubit2 = Qubit(np.array([0, 1]))

    reg_x = Register(np.array([qubit0, qubit1]))
    reg_y = Register(np.array([qubit2]))

    simulation.apply_gates(np.array([H, H]), np.array([0, 1]), reg_x)
    simulation.apply_gates(np.array([H]), np.array([0]), reg_y)

    print(reg_x)
    print(reg_y)

    # Testing
    psi = np.dot(reg_x.reg, reg_y.reg.T)

    f = np.array(
        [
            Qubit(np.array([1, 0])),
            Qubit(np.array([1, 0])),
            Qubit(np.array([0, 1])),
            Qubit(np.array([0, 1])),
        ]
    )

    for val in f:
        print(simulation.apply_CNOT(CNOT_2, val, reg_y.qubits[0], full=False))

    # test_qubit1 = Qubit(np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]))
    # test_qubit2 = Qubit(np.array([1, 0]))
    # test_qubit3 = Qubit(np.array([0, 1]))

    # reg_0 = np.array([test_qubit1, test_qubit2, test_qubit3])
    # test_register0 = Register(reg_0)
    # test_register1 = Register(np.array([]))

    # gates = np.array([H])

    # print(test_register0)
    # print(test_register0.reg)
    # simulation.apply_gates(gates, np.array([2]), test_register0)
    # print(test_register0)
