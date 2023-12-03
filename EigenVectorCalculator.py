import numpy as np
import sys

def calculate_eigen_vector(matrix_str):
    matrix = np.array(eval(matrix_str))
    # print(matrix)
    eig_val, eig_vector = np.linalg.eig(matrix)
    eig_vector = eig_vector / np.linalg.norm(eig_vector)
    # print(eig_vector)
    return eig_vector.tolist()

if __name__ == '__main__':
    matrix_str = sys.stdin.read()
    # matrix_str = "[[1, 2], [3, 4]]"
    eigen_vector = calculate_eigen_vector(matrix_str)
    # t = np.array([2.7445626, 6])
    # print(t/np.linalg.norm(t))
    print(eigen_vector)

