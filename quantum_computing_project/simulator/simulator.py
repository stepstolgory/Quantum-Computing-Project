import numpy as np
from functools import reduce

from quantum_computing_project.register import Register
from quantum_computing_project.gate import Gate
from quantum_computing_project.operations import Operations

class Simulator:

    def __init__(self, registers):
        self.registers = registers
