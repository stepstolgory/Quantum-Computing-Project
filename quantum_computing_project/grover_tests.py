from pickle import FALSE

import scipy.sparse as sps
from quantum_computing_project.operations import Operations
from quantum_computing_project.simulator import Simulator
from quantum_computing_project.register import Register
from quantum_computing_project.constants import *
import matplotlib.pyplot as plt


if __name__ == "__main__":
    """
    Creating a completely random search space of length 3 to 128
    Generate random targets sampled from elements within the search space
    Implements and check to verify that Grover's algorithm is able to produce the same results as classical search
    """
    list_length = np.random.randint(3, 128)
    random_list = list(np.random.randint(1,128, list_length))
    search_length = np.random.randint(1,int(list_length/4))
    random_search = list(np.random.choice(random_list, search_length, replace=False))

    ind_solutions, mods = Simulator.grover_initialise(random_list, random_search, False)
    classical_sol = Simulator.classical_search(random_list, random_search)

    # this checks all the states Grover's algorithm has found by taking states with higher probability than the average
    prob_limit = np.mean(ind_solutions)
    found_states = []
    for i in range(0, len(ind_solutions)):
        if ind_solutions[i] >= prob_limit:
            found_states.append(i)

    if set(found_states) == set(classical_sol):
        print("The algorithm matches with classical search!")
    else:
        print("Mason sucks at coding")
        print(found_states)
        print(classical_sol)

    indices = list(range(len(random_list)))
    plt.bar(indices, ind_solutions, color='skyblue', edgecolor='black')
    plt.xlabel("Indices")
    plt.ylabel("Probability")
    plt.title("Plot of random search list and random search targets")
    #plt.show()


    # checks for "iters" number of randomly generated scenarios
    iters = 1000
    count = 0
    for i in range(0,iters):
        list_length = np.random.randint(8, 128)
        random_list = list(np.random.randint(1, 128, list_length))
        search_length = np.random.randint(1, int(list_length/4))
        random_search = list(np.random.choice(random_list, search_length, replace=False))

        ind_solutions, mods = Simulator.grover_initialise(random_list, random_search, False)
        classical_sol = Simulator.classical_search(random_list, random_search)

        prob_limit = np.mean(ind_solutions)
        found_states = []
        for i in range(0, len(ind_solutions)):
            if ind_solutions[i] >= prob_limit:
                found_states.append(i)

        if set(found_states) == set(classical_sol):
            count += 1

    print(f"Out of {iters} randomly generated scenarios, Grover's algorithm matches with the classical search"
          f" {count} times, representing an accuracy of {(count/iters)*100}%")