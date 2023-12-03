# Code adapted from chat-gpt
import numpy as np
import sys

def calculate_eigen_values(matrix_str):
    matrix = np.array(eval(matrix_str))
    eigen_values = np.linalg.eigvals(matrix)
    return eigen_values.tolist()

if __name__ == '__main__':
    matrix_str = sys.stdin.read()
    eigen_values = calculate_eigen_values(matrix_str)
    print(eigen_values)