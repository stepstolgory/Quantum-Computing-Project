import numpy as np
import scipy.sparse as sps

class Simulator:

    def __init__(self, registers):
        self.registers = registers