import numpy as np
from functools import reduce
import sys
sys.path.append("C:/Users/Sean/Desktop/QCP")
import matplotlib.pyplot as plt
from quantum_computing_project.register import Register
from quantum_computing_project.gate import Gate
from quantum_computing_project.operations import Operations
from quantum_computing_project.constants import *
from quantum_computing_project.grovers_functions import Simulator

input = [16,12,2,3,6,13,4,5,7,9,11,8,1,10,15,14]
ind = np.linspace(0, len(input)-1, len(input))
search_variable = 8
print(Simulator.grover_simplified(input, search_variable))
plt.bar(ind, Simulator.grover_simplified(input, search_variable), color='skyblue', edgecolor='black')
plt.xlabel("State")
plt.ylabel("Probability")
plt.show()
