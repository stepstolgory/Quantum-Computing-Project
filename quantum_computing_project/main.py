from pickle import FALSE

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

def sinusoidal_function():

    """Optional function to plot the sinusoidal function for range of t"""
    input = [14,7,3,1,9,8,13,5,12,2,11,4,10,6]
    search_variables = [12,3,11]
    N = len(input)
    n_sols = len(search_variables)
    theta = np.arcsin(np.sqrt(n_sols / N))
    ideal_t = np.floor(np.pi / (4 * theta))
    print(f"Ideal number of Grover iterations: {ideal_t}")

    t_multiples = np.arange(1, 101)
    print(t_multiples)
    probabilities = []

    # Evaluate probabilities for each t
    ts = t_multiples * ideal_t
    #print(ts)
    # Scale max_t by the current multiple and round to nearest integer
    for t in ts:
        prob_dist = Simulator.grover_calculate(input, search_variables, t)  # Call grover_mod with the computed t

        # Get the probability of the correct solution (first item in search_object)
        correct_solution_index = input.index(search_variables[0])  # Find index of the correct solution in func
        prob_correct_solution = prob_dist[correct_solution_index]  # Probability of that specific solution
        probabilities.append(prob_correct_solution)
    #print(f"Value for t = {ideal_t}: {Simulator.grover_calculate(input, search_variables, ideal_t)}")
    plt.plot(ts, probabilities, marker='o', linestyle='-', color='blue')
    plt.xlabel("Value of t")
    plt.ylabel("Probability of Finding a Solution")
    plt.title("Grover Iterations vs. Solution Probability")
    plt.grid(True)
    plt.show()




if __name__ == "__main__":
    main()
    #sinusoidal_function()
    test_f_balanced = [ZERO, ONE, ONE, ZERO, ONE, ZERO, ZERO, ONE]
    print(f"The function is {'balanced' if Simulator.deutsch_josza(test_f_balanced, 3) else 'constant'}")
    #TODO make the code robust such that it can handle inputs that are not of length that is a power of 2


    new_input = [3,6,4,5,7,8,9,11,12,13,14,8,12]
    search = [3,12,5]

    sol_final, mod_input = Simulator.grover_initialise(new_input, search, False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: States vs Probability
    axes[0].bar(mod_input, sol_final, color="lightcoral", edgecolor="black")
    axes[0].set_xlabel("State")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("State vs. Probability")
    
    # Right plot: Index vs Probability
    indices = list(range(len(mod_input)))
    axes[1].bar(indices, sol_final, color="skyblue", edgecolor="black")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Index vs. Probability")

    plt.tight_layout()
    plt.show()







