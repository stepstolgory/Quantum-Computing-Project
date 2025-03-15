from quantum_computing_project.gate import Gate
import numpy as np
import scipy.sparse as sps
"""
This file contains constants (gates & states) to be used throughout the project.
"""

# Quantum gates
I = Gate(np.array([1, 1]), np.array([0, 1]), np.array([0, 1]), True)

X = Gate(np.array([1, 1]), np.array([0, 1]), np.array([1, 0]), True)

Y = Gate(np.array([-1j, 1j]), np.array([0, 1]), np.array([1, 0]), True)

Z = Gate(np.array([1, -1]), np.array([0, 1]), np.array([0, 1]), True)

S = Gate(np.array([1, 1j]), np.array([0,1]), np.array([0,1]), True)

H = Gate(np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2), -1 / np.sqrt(2)]),
         np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), True)

T = Gate(np.array([1, 1/np.sqrt(2)+1j/np.sqrt(2)]),
         np.array([0, 1]), np.array([0, 1]), True)

CNOT_2 = Gate(np.array([1, 1, 1, 1]),
            np.array([0, 1, 2, 3]), np.array([0, 1, 3, 2]), True)

# States
ZERO = sps.coo_matrix(([1], ([0], [0])), shape=(2, 1))

ONE = sps.coo_matrix(([1], ([1], [0])), shape=(2, 1))

MINUS = sps.coo_matrix((np.array([1/np.sqrt(2), -1/np.sqrt(2)]),
                        (np.array([0, 1]), np.array([0, 0]))), shape=(2, 1))

PLUS = sps.coo_matrix((np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                             (np.array([0, 1]), np.array([0, 0]))), shape=(2, 1))

MINUS_I = sps.coo_matrix((np.array([1/np.sqrt(2), -1j/np.sqrt(2)]),
                          (np.array([0, 1]), np.array([0, 0]))), shape=(2, 1))

PLUS_I = sps.coo_matrix((np.array([1/np.sqrt(2), 1j/np.sqrt(2)]),
                         (np.array([0, 1]), np.array([0, 0]))), shape=(2, 1))

BELL_PHI_PLUS = sps.coo_matrix((np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                                (np.array([0, 3]), np.array([0, 0]))), shape=(4, 1))