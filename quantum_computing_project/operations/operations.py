import scipy.sparse as sps

class Operations:

    @staticmethod
    def tensor(A, B):
        """
        Computes the tensor product between two matrices.

        Args:
            A (numpy.array): 2D array which represents the matrix on the right of the tensor product.
            B (numpy.array): 2D array which represents the matrix on the left of the tensor product.

        Returns:
            np.array: 2D array which represents the matrix resulting from the tensor product A(X)B.
        """
        m, n = A.shape
        p, q = B.shape

        A_flat = A.flatten()
        B_flat = B.flatten()

        outer = A_flat[:, None] * B_flat[None, :]
        outer4d = outer.reshape(m, n, p, q)
        outermpnq = outer4d.transpose(0, 2, 1, 3)
        result = outermpnq.reshape(m * p, n * q)

        return result

    @staticmethod
    def sparse_tensor(A, B):
        """
        Computes the tensor product between two sparse matrices.

        Args:
            A (numpy.array): 2D array which represents the sparse matrix on the right of the tensor product.
            B (numpy.array): 2D array which represents the sparse matrix on the left of the tensor product.

        Returns:
            coo_matrix: 2D array which represents the sparse matrix resulting from the tensor product A(X)B.
        """
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

        C = sps.coo_matrix((new_data, (new_row, new_col)), shape=(m * p, n * q))

        return C

    @staticmethod
    def power_tensor(A, p):
        """
        Performs the tensor product p times between the same matrix.
        A ^ (X)p = A (X) A (X) A (x) A ... (X) A for p matrices.

        Args:
            A (numpy.array): 2D array which represents the input matrix.
            p (int): Number of matrices between which the power tensor is perfomed (p-1 operations).

        Returns:
            numpy.array: 2D array which represents the final result of the operations.
        """
        inter_mat = A
        # start_time = time.time()
        for _ in range(p - 1):
            inter_mat = Operations.tensor(A, inter_mat)
        # end_time = time.time()
        # print(f"Time taken: {end_time-start_time}.")
        return inter_mat

    @staticmethod
    def sparse_power_tensor(A, p):
        """
        Performs the tensor product p times between the same sparse matrix.
        A ^ (X)p = A (X) A (X) A (x) A ... (X) A for p matrices.

        Args:
            A (numpy.array): 2D array which represents the sparse input matrix.
            p (int): Number of sparse matrices between which the power tensor is performed (p-1 operations).

        Returns:
            coo_matrix: sparse 2D array which represents the final result of the operations.
        """
        inter_mat = A
        # start_time = time.time()
        for _ in range(p - 1):
            inter_mat = Operations.sparse_tensor(A, inter_mat)
        # end_time = time.time()
        # print(f"Time taken: {end_time-start_time}.")
        return inter_mat