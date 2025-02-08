from functools import reduce
import numpy as np
import scipy.sparse as sps
from quantum_computing_project.simulator import Simulator

class Register(Simulator):
    """
    A class that represents a quantum register.

    Attributes:
        reg_id (int): A unique identifier for the quantum register.
        n_qubits (int): The number of qubits in the register.
        _reg (numpy.ndarray): The state vector representing the quantum state of the register.
    
    Methods:
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
                           The states should be provided in the form of numpy arrays.
        """
        self.reg_id = Register.reg_id
        Register.reg_id += 1

        self.n_qubits = n_qubits

        if self.n_qubits == 0:
            self._reg = None
        else:
            # self._reg = reduce(self.tensor, states)
            self._reg = reduce(self.sparse_tensor, states)

        Simulator.registers.append(self)
    
    @property
    def reg(self):
        """
        Gets the quantum state of the register.

        Returns:
            numpy.ndarray: The state vector of the quantum register.
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
        Returns the Dirac representation of the quantum register as a string.

        The representation expresses the quantum register's state as a linear combination of basis states,
        with the corresponding complex coefficients rounded to 6 decimal places.

        Returns:
            dr (str): The Dirac representation of the quantum register.
        """
        dr = f"{np.around(self.reg[0, 0], decimals=6)}|0>"
        for i in range(1, 2**self.n_qubits):
            dr += f" + {np.around(self.reg[i, 0], decimals = 6)}|{i}>"
        return dr
    
    def __str__(self):
        """
        Returns a string representation of the quantum register, including its state.

        If the quantum state is not initialised (i.e. no qubits), it returns a message indicating the 
        register contains no qubits. Otherwise, it returns the Dirac representation of the quantum register.

        Returns:
            str: A descriptive string representation of the quantum register.
        """
        try:
            self.dirac_representation
            return f"Register {self.reg_id} contains {self.n_qubits} qubits with representation of\n {self.dirac_representation}"
        except TypeError:
            return f"Register {self.reg_id} contains no qubits!!!"
        
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
        # zero = np.array([[1], [0]])
        zero = sps.coo_matrix(([1], ([0], [0])), shape=(2, 1))
        if isinstance(other, Register):
            new_n_bits = self.n_qubits + other.n_qubits
            newReg = Register(new_n_bits, [zero for _ in range(new_n_bits)])
            # newReg.reg = self.tensor(self.reg, other.reg)
            newReg.reg = self.sparse_tensor(self.reg, other.reg)
            return newReg
        else:
            raise TypeError("The objects added must be instances of the same class.")
    
