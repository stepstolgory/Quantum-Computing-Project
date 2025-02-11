import scipy.sparse as sps
from quantum_computing_project.operations import Operations
from quantum_computing_project.simulator import Simulator
from quantum_computing_project.register import Register
from quantum_computing_project.constants import *

def main():
    myReg = Register(2, [PLUS, ZERO])
    print(myReg)
    myReg.reg = myReg.reg.power(2)

    sample = [i for i in range(myReg.reg.shape[0])]
    probabilities = [p[0] for p in myReg.reg.toarray()]
    print(sample)
    print(probabilities)
    for i in range(10):
        print(f'Result {i}: ', np.random.choice(sample, p=probabilities))


if __name__ == "__main__":
    main()