from numpy import linalg as LA
import numpy as np

def calculate_eigenvalues(matrix: list[list[float]]) -> list[float]:

    eigenvalues, eigenvectors = LA.eig(np.array(matrix))
    return eigenvalues


a = [[2.0, 1.0], [1.0, 2.0]]
b = [[4.0, -2.0], [1.0, 1.0]]

print("Output: ", calculate_eigenvalues(a))
print("Output: ", calculate_eigenvalues(b))
