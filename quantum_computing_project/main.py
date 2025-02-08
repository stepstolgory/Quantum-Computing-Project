import numpy as np
import scipy.sparse as sps
from quantum_computing_project import Operations
from quantum_computing_project import Simulator
from quantum_computing_project import Register
from quantum_computing_project import I, X

def main():
    zero = sps.coo_matrix(([1], ([0], [0])), shape=(2, 1))

    myGates = np.array([X(), I()])
    myReg = Register(2, [zero, zero])
    print('start: ', myReg.reg.toarray())

    sim = Simulator([myReg])
    sim.apply_gates(myGates, myReg)
    print('end: ', myReg.reg.toarray())


if __name__ == "__main__":
    main()