import numpy as np
import matplotlib.pyplot as plt

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:

    n = len(vectors[0]) # obs
    k = len(vectors) # var
    print(n)

    med = []
    for var in vectors:
        med.append(sum(var) / n)

    cov = []

    for i in range(k):
        lin = []
        for j in range(k):
            summ = 0
            for obs in range(n):
                value_i = vectors[i][obs] - med[i]
                value_j = vectors[j][obs] - med[j]
                prod = value_i * value_j
                summ += prod


            covs = summ / (n-1)
            lin.append(covs)

        cov.append(lin)

    return cov

input: list[list[float]] = [[1, 2, 3], [4, 5, 6]]

print(calculate_covariance_matrix(input))
