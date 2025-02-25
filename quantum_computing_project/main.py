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
    #TODO make the code robust such that it can handle inputs that are not of length that is a power of 2
    input = [ 16,12, 2, 3, 6, 13, 4, 5, 7, 9, 11, 8, 1, 10, 15, 14]
    ind = np.linspace(0, len(input) - 1, len(input))
    search_variable = 8

    # print(Simulator.grover_simplified(input, search_variable))
    #plt.bar(ind, Simulator.grover_simplified(input, search_variable), color='skyblue', edgecolor='black')
    #Think the plot of the solutions vs probability might be better visualisation than index vs probability. Can discuss further.
    #plt.bar(input, Simulator.grover_simplified(input, search_variable), color='skyblue', edgecolor='black')
    #plt.xlabel("State")
    #plt.ylabel("Probability")
    #plt.show()

    search_variables = [16]
    y = Simulator.grover_multiple_known_sols(input,search_variables)
    plt.bar(input,y, color = "red", edgecolor = "black")
    plt.xlabel("State")
    plt.ylabel("Probability")
    plt.show()

    #this is for plotting probs of sol vs t
    N = len(input)
    n_sols = len(search_variables)
    theta = np.arcsin(np.sqrt(n_sols / N))
    ideal_t = np.floor(np.pi / (4 * theta))
    print(f"Ideal number of Grover iterations: {ideal_t}")

    t_multiples = np.linspace(0, 100, 100)
    print(t_multiples)
    probabilities = []

    # Evaluate probabilities for each t
    ts = t_multiples * ideal_t
    print(ts)# Scale max_t by the current multiple and round to nearest integer
    for t in ts:
        prob_dist = Simulator.grover_mod(input, search_variables, t)  # Call grover_mod with the computed t

        # Get the probability of the correct solution (first item in search_object)
        correct_solution_index = input.index(search_variables[0])  # Find index of the correct solution in func
        prob_correct_solution = prob_dist[correct_solution_index]  # Probability of that specific solution
        probabilities.append(prob_correct_solution)
    print(f"Value for t = {ideal_t}: {Simulator.grover_mod(input, search_variables, ideal_t)}")
    plt.plot(ts, probabilities, marker='o', linestyle='-', color='blue')
    plt.xlabel("Value of t")
    plt.ylabel("Probability of Finding a Solution")
    plt.title("Grover Iterations vs. Solution Probability")
    plt.grid(True)
    plt.show()


