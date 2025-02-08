import numpy as np
from quantum_computing_project import Register

def main():
    zero = np.array([[1], [0]])
    one = np.array([[0], [1]]) 

    reg = Register(2, [zero, zero]) + Register(3, [zero, zero, zero])
    print(reg._reg == Register(5, [zero, zero, zero, zero, zero])._reg)


if __name__ == "__main__":
    main()