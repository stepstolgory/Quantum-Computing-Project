import numpy as np
from functools import reduce

from quantum_computing_project.register import Register
from quantum_computing_project.gate import Gate
from quantum_computing_project.operations import Operations

class Simulator:
    """
    The Simulator class is where Deutsch-Josza, Grover's etc. will be implemented.
    """

    def __init__(self, registers):
        self.registers = registers

    def measure(self, register: Register):
        ...