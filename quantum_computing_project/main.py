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
    Simulator.nine_qubit_shor(phase_flip=True, bit_flip=False)


if __name__ == "__main__":
    main()
    # test_f_balanced = [ZERO, ONE, ONE, ZERO, ONE, ZERO, ZERO, ONE]
    # print(f"The function is {'balanced' if Simulator.deutsch_jozsa(test_f_balanced, 3) else 'constant'}")
    #
    # input = [16, 12, 2, 3, 6, 13, 4, 5, 7, 9, 11, 8, 1, 10, 15, 14]
    # ind = np.linspace(0, len(input) - 1, len(input))
    # search_variable = 8
    #
    # print(Simulator.grover_simplified(input, search_variable))
    # plt.bar(ind, Simulator.grover_simplified(input, search_variable), color='skyblue', edgecolor='black')
    # plt.xlabel("State")
    # plt.ylabel("Probability")
    # plt.show()