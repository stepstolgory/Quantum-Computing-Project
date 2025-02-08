import numpy as np
import scipy.sparse as sps
from quantum_computing_project import Operations

def main():
    data_s = np.array([1, 1j])
    row_s = np.array([0, 1])
    col_s = np.array([0, 1])
    s_sparse = sps.coo_matrix((data_s, (row_s, col_s)))

    result = Operations.sparse_power_tensor(s_sparse, 2)
    expected_data = np.array([1, 1j, 1j, -1])
    expected_row = np.array([0, 1, 2, 3])
    expected_col = np.array([0, 1, 2, 3])
    expected = sps.coo_matrix((expected_data, (expected_row, expected_col)))
    #print(result)
    #print(expected)

    s_arr = np.array([[1, 0], [0, 1j]])
    result_arr = Operations.power_tensor(s_arr, 2)
    expected_arr = np.array([[1, 0, 0, 0],
                                [0, 1j, 0, 0],
                                [0, 0, 1j, 0],
                                [0, 0, 0, -1]])
    #print(result_arr)
    #print(expected_arr)
    print(expected_arr == result.toarray())

if __name__ == "__main__":
    main()