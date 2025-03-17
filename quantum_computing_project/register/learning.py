import scipy.sparse as sps
from quantum_computing_project.operations import Operations
from quantum_computing_project.simulator import Simulator
from quantum_computing_project.register import Register
from quantum_computing_project.constants import *
import matplotlib.pyplot as plt

print(ZERO)
print(ONE)
print(PLUS)
print(MINUS)

first_register = Register(5, [ONE, ONE, ONE, ZERO, ONE])
# print(first_register.reg)
N = 10
grover_reg = Register(N,[ZERO for i in range(0,N)])
grover_reg.apply_gates(np.array([H]).repeat(grover_reg.n_qubits))
print(grover_reg.reg)
