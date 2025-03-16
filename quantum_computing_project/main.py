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
    comb_2.apply_gates(np.array([H, H, H]))

    # APPLY X ERROR
    comb_2.apply_gates(np.array([Z, I, I])) # -- flips last qubit

    # DETECTING
    # Set up registers
    comb_2.apply_gates(np.array([H, H, H]))
    reg_2a = Register(1, [ZERO])
    reg_2b = Register(1, [ZERO])
    system = comb_2 + reg_2a + reg_2b

    # State of system is now |00100> + |11000>
    # Apply gates
    system.reg = np.dot(CNOT_14.gate, system.reg).tocoo()
    system.reg = np.dot(CNOT_24.gate, system.reg).tocoo()
    system.reg = np.dot(CNOT_25.gate, system.reg).tocoo()
    system.reg = np.dot(CNOT_35.gate, system.reg).tocoo()

    # State of system is now |00101> + |11001> -- correct so far!
    # Measurement
    b_syndrome = measure_register_n(system.reg, 4)
    c_syndrome = measure_register_n(system.reg, 5)
    result = (int(b_syndrome), int(c_syndrome))
    print(result)

def measure_register_n(state, n):
    """
    Measures the n-th qubit in a 5-qubit state.

    The function computes the total probability for the fourth qubit being 0 or 1.
    It then returns the measurement outcome (0 or 1).
    """
    prob_0 = 0.0
    prob_1 = 0.0

    # Iterate over the nonzero elements in the state vector
    for idx, amp in zip(state.row, state.data):
        # Convert the index to a 5-bit binary string
        bits = format(idx, '05b')
        # The n-th qubit is at position n-1
        if bits[n-1] == '0':
            prob_0 += np.abs(amp) ** 2
        else:
            prob_1 += np.abs(amp) ** 2

    # Normalise probabilities
    total = prob_0 + prob_1
    if total > 0:
        prob_0 /= total
        prob_1 /= total

    outcome = np.random.choice([0, 1], p=[prob_0, prob_1])
    return outcome

if __name__ == "__main__":
    main()

