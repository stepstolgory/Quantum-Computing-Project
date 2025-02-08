import scipy.sparse as sps
from quantum_computing_project.operations import Operations
from quantum_computing_project.simulator import Simulator
from quantum_computing_project.register import Register
from quantum_computing_project.constants import *

def main():
    myReg = Register(1, [ZERO])
    print(myReg)


if __name__ == "__main__":
    main()