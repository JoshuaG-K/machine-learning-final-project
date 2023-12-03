import numpy as np
import sys

def calculate_eigen_vector(matrix_str):
    matrix = np.array(eval(matrix_str))
    eig_val, eig_vector = np.linalg.eig(matrix)

    return eig_vector.tolist()

if __name__ == '__main__':
    matrix_str = sys.stdin.read()
    eigen_vector = calculate_eigen_vector(matrix_str)
    print(eigen_vector)

