import numpy as np
import scipy.linalg
import scipy.sparse
import time


def normalise_state(state):
    return state / scipy.linalg.norm(state)


def tensor(A, B):
    """A (X) B = each element of Aij*B so n x n (X) n x n gives n^2 x n^2 matrix"""

    m, n = A.shape
    p, q = B.shape

    A_flat = A.flatten()
    B_flat = B.flatten()

    outer = A_flat[:, None] * B_flat[None, :]
    outer4d = outer.reshape(m, n, p, q)
    outermpnq = outer4d.transpose(0, 2, 1, 3)
    result = outermpnq.reshape(m * p, n * q)

    return result


def sparse_tensor(A, B):
    m, n = A.shape
    p, q = B.shape

    new_row = []
    new_col = []
    new_data = []

    for ai, aj, aval in zip(A.row, A.col, A.data):
        for bi, bj, bval in zip(B.row, B.col, B.data):
            row_id = ai * p + bi
            col_id = aj * q + bj
            val = aval * bval

            new_row.append(row_id)
            new_col.append(col_id)
            new_data.append(val)

    C = scipy.sparse.coo_matrix((new_data, (new_row, new_col)), shape=(m * p, n * q))

    return C


def sparse_power_tensor(A, p):
    inter_mat = A
    start_time = time.time()
    for _ in range(p):
        inter_mat = sparse_tensor(A, inter_mat)
    end_time = time.time()
    print(f"Time taken: {end_time-start_time}.")
    return inter_mat


def power_tensor(A, p):
    inter_mat = A
    start_time = time.time()
    for _ in range(p):
        inter_mat = tensor(A, inter_mat)
    end_time = time.time()
    print(f"Time taken: {end_time-start_time}.")
    return inter_mat


zero = np.array([[1], [0]])

one = np.array([[0], [1]])

zero_sparse = scipy.sparse.coo_matrix(([1], ([0], [0])), shape=(2, 1))
one_sparse = scipy.sparse.coo_matrix(([1], ([1], [0])), shape=(2, 1))

H = 1.0 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
S = np.array([[1, 0], [0, 1j]])
phi = lambda phi: np.array([[1, 0], [0, np.exp(phi * 1j)]])
T = phi(np.pi / 4)
CNOT2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


x = zero
y = one

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])

# Trying coo matrices
A_data = np.array([1]).repeat(2)
B_data = np.array([-1j, 1j])
row = np.array([0, 1])
column = np.array([1, 0])
A = scipy.sparse.coo_matrix((A_data, (row, column)), shape=(2, 2))
B = scipy.sparse.coo_matrix((B_data, (row, column)), shape=(2, 2))

try:
    print("^^^ A (X) B result")
except Exception as e:
    print(f"Something went wrong: {e.__class__.__name__}")
print("^^^ X (X) Y result")

H_data = 1 / np.sqrt(2) * np.array([1, 1, 1, -1])
H_row = np.array([0, 0, 1, 1])
H_col = np.array([0, 1, 0, 1])
H_sparse = scipy.sparse.coo_matrix((H_data, (H_row, H_col)), shape=(2, 2))

# Comparison of sparse matrix with no zeros to normal matrix
H_sparse_res = sparse_power_tensor(H_sparse, 10)
H_norm_res = power_tensor(H, 10)

# Comparison of sparse matrix with a lot of zeros to normal matrix
X_sparse_res = sparse_power_tensor(A, 13)
X_norm_res = power_tensor(X, 13)

psi1 = np.array([x, y])
psi2 = np.array([np.dot(H, zero), np.dot(H, one)])
# print(psi1)
# print(psi2)
zerozero = np.kron(zero, zero)
zeroone = np.kron(zero, one)
onezero = np.kron(one, zero)
oneone = np.kron(one, one)
print(zerozero + oneone)
zeroonezero = tensor(tensor(zero, one), zero)
zerozeroone = tensor(tensor(zero, zero), one)


two_ones = sparse_power_tensor(one_sparse, 2)
two_H = power_tensor(H, 2)

print(two_H)
test_application = two_H * two_ones
print(test_application)

# print(np.tensordot(zero, one, axes=0))

psi = normalise_state((zerozero + onezero))
psi2 = np.dot(CNOT2, psi)
# print(psi2)


test_qubit1 = np.dot(H, zero)
test_qubit2 = zero
print(test_qubit1)
print(test_qubit2)
print(tensor(test_qubit2, test_qubit1))
