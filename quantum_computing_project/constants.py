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

ls3 = [i for i in range(6)]
ls3.append(2**3-1)
ls3.append(2**3-2)
CNOT_3 = Gate(np.array([1 for _ in range(8)]),
              np.array([i for i in range(8)]),
              np.array(ls3),True)

# Special CNOTs for 9 qubit Shor code
ls14_1 = [i for i in range(16)]
ls14_2 = [18, 19, 16, 17, 22, 23, 20, 21, 26, 27, 24, 25, 30,  31, 28, 29]
ls14 = ls14_1 + ls14_2
CNOT_14 = Gate(np.array([1 for _ in range(32)]),
              np.array([i for i in range(32)]),
              np.array(ls14),True)

ls24_1 = [i for i in range(8)]
ls24_2 =[10, 11, 8, 9, 14, 15, 12, 13]
ls24_3 = [i for i in range(16,24)]
ls24_4 = [26, 27, 24, 25, 30, 31, 28, 29]
ls24 = ls24_1 + ls24_2 + ls24_3 + ls24_4
CNOT_24 = Gate(np.array([1 for _ in range(32)]),
              np.array([i for i in range(32)]),
              np.array(ls24),True)

ls25_1 = [i for i in range(8)]
ls25_2 =[9, 8, 11, 10, 13, 12, 15, 14]
ls25_3 = [i for i in range(16,24)]
ls25_4 = [25, 24, 27, 26, 29, 28, 31, 30]
ls25 = ls25_1 + ls25_2 + ls25_3 + ls25_4
CNOT_25 = Gate(np.array([1 for _ in range(32)]),
              np.array([i for i in range(32)]),
              np.array(ls25),True)

ls35_1 = [i for i in range(4)]
ls35_2 = [5, 4, 7, 6]
ls35_3 = [i for i in range(8,12)]
ls35_4 = [13, 12, 15, 14]
ls35_5 = [i for i in range(16,20)]
ls35_6 = [21, 20, 23, 22]
ls35_7 = [i for i in range(24, 28)]
ls35_8 = [29, 28, 31, 30]
ls35 = ls35_1 + ls35_2 + ls35_3 + ls35_4 + ls35_5 + ls35_6 +ls35_7 + ls35_8
CNOT_35 = Gate(np.array([1 for _ in range(32)]),
              np.array([i for i in range(32)]),
              np.array(ls35),True)


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