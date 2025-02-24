import scipy.sparse as sps
from quantum_computing_project.operations import Operations
from quantum_computing_project.simulator import Simulator
from quantum_computing_project.register import Register
from quantum_computing_project.constants import *
import matplotlib.pyplot as plt

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
    test_f_balanced = [ZERO, ONE, ONE, ZERO, ONE, ZERO, ZERO, ONE]
    print(f"The function is {'balanced' if Simulator.deutsch_josza(test_f_balanced, 3) else 'constant'}")

    input = [16, 12, 2, 3, 6, 13, 4, 5, 7, 9, 11, 8, 1, 10, 15, 14]
    ind = np.linspace(0, len(input) - 1, len(input))
    search_variable = 8

    print(Simulator.grover_simplified(input, search_variable))
    plt.bar(ind, Simulator.grover_simplified(input, search_variable), color='skyblue', edgecolor='black')
    plt.xlabel("State")
    plt.ylabel("Probability")
    plt.show()