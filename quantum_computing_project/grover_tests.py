from pickle import FALSE

import scipy.sparse as sps
from quantum_computing_project.operations import Operations
from quantum_computing_project.simulator import Simulator
from quantum_computing_project.register import Register
from quantum_computing_project.constants import *
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

if __name__ == "__main__":
    """
    - Creating a completely random search space of length 8 to 128
    - Generate random targets sampled from elements within the search space
    - Implements and check to verify that Grover's algorithm is able to produce the same results as classical search
    - The code then measures the results 100 times and counts how many times the measurements are correct
    - The final part runs 100 randomly generated scenarios to check how many times the marked states are correct in the probability
      distribution
    """

    list_length = np.random.randint(8, 128)
    random_list = list(np.random.randint(1,128, list_length))
    search_length = np.random.randint(1,int(list_length/4))
    random_search = list(np.random.choice(random_list, search_length, replace=False))

    ind_solutions = Simulator.grover_initialise(random_list, random_search, False)[0]
    classical_sol = Simulator.classical_search(random_list, random_search)

    # this determines all the states Grover's algorithm has found by taking states with greater amplitude than the average
    prob_limit = np.mean(ind_solutions)
    found_states = []
    for i in range(0, len(ind_solutions)):
        if ind_solutions[i] >= prob_limit:
            found_states.append(i)

    # if all the marked states are correct then the algorithm has worked
    if set(found_states) == set(classical_sol):
        print("The algorithm matches with classical search!")
    else:
        print("Mason sucks at coding")
        print(found_states)
        print(classical_sol)


    # plots initial probabilities of states of a single randomly generated scenario
    indices = list(range(len(random_list)))
    plt.bar(indices, ind_solutions, color='skyblue', edgecolor='black')
    plt.xlabel("Indices")
    plt.ylabel("Probability")
    plt.title("Plot of random search list and random search targets")


    # checks the percentage of times a measurement results in a desired state for the single randomly generated scenario
    measurement_iters = 100
    measurement_checker = 0
    for i in range(0, measurement_iters):
        measurement = Simulator.grover_initialise(random_list, random_search, False)[1]
        if measurement in classical_sol:
            measurement_checker += 1

    print(f"Out of {measurement_iters} measurements of the same scenario, {measurement_checker} of them matched with"
          f" results from a classical search representing an accuracy of {(measurement_checker/measurement_iters)*100}%")


    # checks the percentage of states are correctly marked in the distribution for 100 randomly generated scenarios
    distribution_iters = 100
    distribution_checker = 0
    for i in range(0,distribution_iters):
        list_length = np.random.randint(8, 128)
        random_list = list(np.random.randint(1, 128, list_length))
        search_length = np.random.randint(1, int(list_length/4))
        random_search = list(np.random.choice(random_list, search_length, replace=False))

        ind_solutions, measurement = Simulator.grover_initialise(random_list, random_search, False)
        classical_sol = Simulator.classical_search(random_list, random_search)

        prob_limit = np.mean(ind_solutions)
        found_states = []
        for i in range(0, len(ind_solutions)):
            if ind_solutions[i] >= prob_limit:
                found_states.append(i)

        if set(found_states) == set(classical_sol):
            distribution_checker += 1

    print(f"Out of {distribution_iters} randomly generated scenarios, Grover's algorithm correctly marked all the states that"
          f" matched with classical search {distribution_checker} times, representing an accuracy of"
          f" {(distribution_checker/distribution_iters)*100}%")

    plt.show()



