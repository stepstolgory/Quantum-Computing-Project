# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 03:01:05 2025

@author: Sean
"""

# Import required Qiskit components and matplotlib for visualization
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Define the number of qubits. We have 2 qubits for our 4-item list.
n_qubits = 2

# Create a quantum circuit with 2 qubits and 2 classical bits (for measurement)
qc = QuantumCircuit(n_qubits, n_qubits)

# Step 1: Initialization
# Apply Hadamard gates to all qubits to create the uniform superposition:
#   |U> = 1/2 (|00> + |01> + |10> + |11>)
qc.h(range(n_qubits))

# Step 2: Oracle Construction
# We want to mark the state |01> (the index corresponding to the number 7).
# To do this:
#   a. Apply an X gate to qubit 0 so that |01> becomes |11>.
#   b. Apply a CZ gate which flips the phase of |11>.
#   c. Apply an X gate again to qubit 0 to revert the mapping.
qc.x(0)
qc.cz(0, 1)
qc.x(0)

# At this point, the oracle O acts as:
#   O|00> = |00>, O|01> = -|01>, O|10> = |10>, O|11> = |11>.

# Step 3: Diffusion Operator (Inversion about the Mean)
# The diffusion operator is given by:
#   D = H⊗2 · (2|00><00| - I) · H⊗2.
# For 2 qubits, this can be implemented as:
qc.h(range(n_qubits))
qc.x(range(n_qubits))
qc.cz(0, 1)
qc.x(range(n_qubits))
qc.h(range(n_qubits))

# Step 4: Measurement
# Measure the qubits into the classical bits.
qc.measure(range(n_qubits), range(n_qubits))

# Draw the circuit
print(qc.draw('mpl'))

# Execute the circuit on the qasm_simulator
simulator = Aer.get_backend('qasm_simulator')
job = execute(qc, simulator, shots=1024)
result = job.result()
counts = result.get_counts(qc)

# Print and plot the measurement results.
print("\nMeasurement Results:", counts)
plot_histogram(counts)
plt.show()
