import scipy.sparse as sps
from quantum_computing_project.operations import Operations
from quantum_computing_project.simulator import Simulator
from quantum_computing_project.register import Register
from quantum_computing_project.constants import *
import matplotlib.pyplot as plt
import time
import tracemalloc

def dj_performance(n_qubits):
    func_len = 2**n_qubits
    f_constant = []
    for i in range(func_len):
        f_constant.append(ZERO)
    start = time.time()
    result = Simulator.deutsch_jozsa(f_constant, n_qubits)
    end = time.time()
    time_taken = end - start
    # print("Balanced? " + str(result) + "\n Time taken: " + str(time_taken)
          #+ "\n Qubits: " + str(n_qubits))
    return time_taken

def dj_time_performance(max_qubits):
    """DJ Performance: Num Qubits vs Time Taken"""
    x_qubits = np.array([])
    y_time = np.array([])
    for i in range(1, max_qubits+1):
        time_taken= dj_performance(i)
        x_qubits = np.append(x_qubits, i)
        y_time = np.append(y_time, time_taken)
    plt.plot(x_qubits, y_time)
    plt.xlabel("Number of qubits")
    plt.ylabel("Time (seconds)")
    plt.show()

def dj_memory_performance(max_qubits):
    """DJ Performance: Num Qubits vs Memory Required"""
    x_qubits = np.array([])
    y_peak = np.array([])
    for i in range(1, max_qubits+1):
        tracemalloc.start()
        dj_performance(i)
        current, peak = tracemalloc.get_traced_memory()
        # print(f"Current memory usage: {(current / 1024**2):.2f} MB")
        # print(f"Peak memory usage: {(peak / 1024**2):.2f} MB")
        tracemalloc.stop()
        x_qubits = np.append(x_qubits, i)
        y_peak = np.append(y_peak, peak / 1024 ** 2)
    plt.plot(x_qubits, y_peak)
    plt.xlabel("Number of qubits")
    plt.ylabel("Peak memory usage (MB)")
    plt.show()

def main():
    # dj_time_performance(12)
    # dj_memory_performance(12)

    # ENCODING
    # Set up registers
    psi = Register(1, [PLUS])
    reg_1a = Register(1, [ZERO])
    reg_1b = Register(1, [ZERO])

    # Apply gates
    comb_1 = reg_1a.apply_CNOT_sup(psi, dim = 2)
    comb_2 = reg_1b.apply_CNOT_sup(comb_1, dim = 3)
    # print(comb_2.reg.toarray()) # -- correct so far!

    # APPLY X ERROR
    comb_2.apply_gates(np.array([I, I, X])) # -- flips last qubit

    # DETECTING
    # Set up registers
    reg_2a = Register(1, [ZERO])
    reg_2b = Register(1, [ZERO])
    system = comb_2 + reg_2a + reg_2b

    # State of comb_4 is now |00100> + |11000>
    # Apply gates


def t():
    # |000> + |111>
    a1 = Operations.sparse_tensor(ZERO, ZERO)
    a2 = Operations.sparse_tensor(a1, ZERO)
    a3 = Operations.sparse_tensor(ONE, ONE)
    a4 = Operations.sparse_tensor(a3, ONE)
    A = 1/np.sqrt(2)*(a2+a4)
    A = A.tocoo()

    # |100> + |011>
    b1 = Operations.sparse_tensor(ONE, ZERO)
    b2 = Operations.sparse_tensor(b1, ZERO)
    b3 = Operations.sparse_tensor(ZERO, ONE)
    b4 = Operations.sparse_tensor(b3, ONE)
    B = 1 / np.sqrt(2) * (b2 + b4)
    B = B.tocoo()

    # |010> + |101>
    c1 = Operations.sparse_tensor(ZERO, ONE)
    c2 = Operations.sparse_tensor(c1, ZERO)
    c3 = Operations.sparse_tensor(ONE, ZERO)
    c4 = Operations.sparse_tensor(c3, ONE)
    C = 1 / np.sqrt(2) * (c2 + c4)
    C = C.tocoo()

    # |001> + |110>
    d1 = Operations.sparse_tensor(ZERO, ZERO)
    d2 = Operations.sparse_tensor(d1, ONE)
    d3 = Operations.sparse_tensor(ONE, ONE)
    d4 = Operations.sparse_tensor(d3, ZERO)
    D = 1 / np.sqrt(2) * (d2 + d4)
    D = D.tocoo()

    A_res = Operations.sparse_tensor(A, ZERO)
    B_res = Operations.sparse_tensor(B, ZERO)
    C_res = Operations.sparse_tensor(C, ZERO)
    D_res = Operations.sparse_tensor(D, ZERO)

    print(D)


if __name__ == "__main__":
    # |01000>
    x = Register(5, [ZERO, ONE, ZERO, ZERO, ZERO])
    result = np.dot(CNOT_24.gate, x.reg)
    print(result)

    # |01010>
    y = Register(5, [ZERO, ONE, ZERO, ONE, ZERO])
    print(y.reg)
