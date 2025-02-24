from functools import reduce
import numpy as np
import scipy.sparse as sps
from quantum_computing_project.operations import Operations
from quantum_computing_project.gate import Gate
from quantum_computing_project import constants

class Register:
    """
    A class that represents a quantum register.

    Attributes:
        reg_id (int): A unique identifier for the quantum register.
        n_qubits (int): The number of qubits in the register.
        _reg (numpy.ndarray): The state vector representing the quantum state of the register.
    
    Methods:
        apply_gates: Applies n gates to a register with n qubits.
        reg (property): Getter for the quantum state of the register.
        reg (setter): Setter for the quantum state of the register.
        dirac_representation (property): Returns the Dirac representation of the quantum register.
        __str__(): Returns a string representation of the quantum register, including its state.
        __add__(other): Allows for the concatenation of two registers into a new register.
    """
    reg_id = 0

    def __init__(self, n_qubits, states):
        """
        Initialises a quantum register with a specified number of qubits and quantum states.

        Args:
            n_qubits (int): The number of qubits in the quantum register.
            states (list): A list of states that initialise the quantum register. 
                           The states should be provided in the form of coo_matrices.
        """
        self.reg_id = Register.reg_id
        Register.reg_id += 1

        self.n_qubits = n_qubits

        if self.n_qubits == 0:
            self._reg = None
        else:
            # self._reg = reduce(self.tensor, states)
            self._reg = reduce(Operations.sparse_tensor, states)

    def apply_gates(self, gates):
        """
        Applies n gates on a register with n qubits.

        Args:
            gates (numpy.array of Gate objects): The gates that are applied to the register. The array must
                                                 be of the same length as the number of qubits in the register.

        Raises:
            ValueError: Makes sure the number of gates matches the number of qubits in the register.
            TypeError: Makes sure that the `gates` variable is of the correct type.
        """
        if all(isinstance(g, Gate) for g in gates):

            gates = np.array([g.gate for g in gates])
            resulting_gate = reduce(Operations.sparse_tensor, gates)
            if resulting_gate.shape[1] != self.reg.shape[0]:
                raise ValueError(
                    "The size of the gate must match the size of the register"
                )
            else:
                self.reg = (np.dot(resulting_gate, self.reg)).tocoo()
        else:
            raise TypeError(
             "The 'gates' parameter must be a numpy.ndarray of Gate objects."
         )

    def apply_CNOT(self, control_qubit):
        """Applies a CNOT gate to a register consisting of a single qubit

        Args:
            CNOT (Gate): An instance of the Gate class which represents the 2 qubit CNOT gate
            control_qubit (numpy.array): State of the qubit that controls the gate. (zero or one)
            register (Register): An instance of the Register class which represents a register with a single qubit that will be acted on by the CNOT gate

        Returns:
            numpy.array: The array of the separate amplitudes of the zero state and the one state of the second (original) qubit.
        """
        reg_with_control = Operations.sparse_tensor(control_qubit, self.reg)
        final_reg = np.dot(constants.CNOT_2.gate, reg_with_control)

        # Factor out the second qubit
        zero_amp = np.sum(final_reg[0::2])
        one_amp = np.sum(final_reg[1::2])

        # var.qubit = np.array([[zero_amp], [one_amp]])
        resulting_amps = np.array([[zero_amp], [one_amp]])

        return resulting_amps

    def measure(self):
        """
        Simulates a quantum measurement by collapsing the register's state.

        The method squares the state vector elements to obtain probabilities,
        then samples an index based on this distribution.

        Returns:
            int: The measured state index.
        """
        self.reg = self.reg.power(2)
        sample = [i for i in range(self.reg.shape[0])]
        probabilities = [p[0] for p in self.reg.toarray()]
        return np.random.choice(sample, p=probabilities)

    def distribution(self):
        """
        Simulates a quantum measurement by collapsing the register's state.

        The method squares the state vector elements to obtain probabilities,
        then samples an index based on this distribution.

        Returns:
            int: The measured state index.
        """
        self.reg = self.reg.power(2)
        sample = [i for i in range(self.reg.shape[0])]
        probabilities = [p[0] for p in self.reg.toarray()]
        return probabilities

    @property
    def reg(self):
        """
        Gets the quantum state of the register.

        Returns:
            sps.coo_matrix: The state vector of the quantum register.
        """
        return self._reg

    @reg.setter
    def reg(self, value):
        """
        Sets the quantum state of the register.

        Args:
            value (numpy.ndarray): The state vector to set for the register.
        """
        self._reg = value
    
    @property
    def dirac_representation(self):
        """
        Returns the Dirac representation of the quantum register's state as a string.

        Returns:
            dr (str): The Dirac representation of the quantum register.
        """
        dense_reg = self.reg.toarray()
        dr = f"{np.around(dense_reg[0, 0], decimals=6)}|0⟩"
        for i in range(1, 2**self.n_qubits):
            dr += f" + {np.around(dense_reg[i, 0], decimals = 6)}|{i}⟩"
        return dr
    
    def __str__(self):
        """
        Returns a string representation of the quantum register, including its state. If the
        register has no qubits, a string saying so is returned.

        Returns:
            str: The string representation of the quantum register.
        """
        if self.reg_id is None:
            return f"Register {self.reg_id} contains no qubits!!!"
        else:
            return f"Register {self.reg_id} contains {self.n_qubits} qubits with representation of\n {self.dirac_representation}"
        # try:
        #     self.dirac_representation
        #     return f"Register {self.reg_id} contains {self.n_qubits} qubits with representation of\n {self.dirac_representation}"
        # except TypeError:
        #     return f"Register {self.reg_id} contains no qubits!!!"
        
    def __add__(self, other):
        """
        Concatenates two quantum registers to create a new one.

        Args:
            other (Register): Another quantum register to be concatenated with the current register.

        Returns:
            Register: A new quantum register containing the concatenated states of both registers.

        Raises:
            TypeError: If the other object is not an instance of the Register class.
        """
        # zero = np.array([[1], [0]])
        if isinstance(other, Register):
            new_n_bits = self.n_qubits + other.n_qubits
            # newReg = Register(new_n_bits, [zero for _ in range(new_n_bits)])
            newReg = Register(new_n_bits, [constants.ZERO for _ in range(new_n_bits)])
            # newReg.reg = self.tensor(self.reg, other.reg)
            newReg.reg = Operations.sparse_tensor(self.reg, other.reg)
            return newReg
        else:
            raise TypeError("The objects added must be instances of the same class.")
    
