import scipy.sparse as sps
from quantum_computing_project.operations import Operations
from quantum_computing_project.simulator import Simulator
from quantum_computing_project.register import Register
from quantum_computing_project.constants import *

def main():
    myGates = np.array([I, X])
    myReg = Register(2, [ZERO, ONE])
    print('start: ', myReg.reg.toarray())

    sim = Simulator([myReg])
    sim.apply_gates(myGates, myReg)
    print('end: ', myReg.reg.toarray())


if __name__ == "__main__":
    main()