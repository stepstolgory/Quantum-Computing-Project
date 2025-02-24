import scipy.sparse as sps
import sys
sys.path.append("C:/Users/Sean/Desktop/QCP")
from quantum_computing_project.operations import Operations
#from quantum_computing_project_2.simulator import Simulator
from quantum_computing_project.register import Register
from quantum_computing_project.constants import *
from quantum_computing_project.grovers_functions import Simulator

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


    n_inputs = 2
    grover_state = Register(n_inputs, [ZERO for _ in range(n_inputs)])
    grover_state.apply_gates(np.array([H]).repeat(grover_state.n_qubits))
    R0_data = -np.ones(2**grover_state.n_qubits)
    R0_data[0] = 1
    R0 = Gate(R0_data, np.linspace(0, (2**n_inputs)-1, 2**n_inputs, dtype=int), np.linspace(0, (2**n_inputs)-1, 2**n_inputs, dtype=int), True)
    grover_state.apply_gates(np.array([H]).repeat(grover_state.n_qubits))
    grover_state.apply_gates(np.array([R0]))
    grover_state.apply_gates(np.array([H]).repeat(grover_state.n_qubits))
    
    print(grover_state)
    print(grover_state._reg)

    """ 
    test_gate = Gate(np.array([1,1,1,1]), np.array([0,0,0,0]), np.array([0,1,2,3]), False)
    grover_state.apply_gates(np.array([test_gate]))
    print(grover_state)
    
    def identity(n):
        I = Gate(np.ones(n), np.linspace(0, n-1, n).astype(int), np.linspace(0, n-1, n).astype(int), True)
        return I.gate
    print(identity(10))

    reg_test = Register(2, [ZERO for _ in range(2)])
    reg_test.apply_gates(np.array([H]).repeat(2))

    print(reg_test._reg)
    U_2d = reg_test._reg.toarray()
    U_outer = np.outer(U_2d, U_2d)
    print(U_outer)
    """
    