import numpy as np

x_i = 0.4
x_j = 0.4
sigma_i = 1.0 # ou perplexidade

# p0,j = e^(-( ||x0 - xj||^2 ) / (2 * σ0^2))
p_ij = np.exp(-np.linalg.norm(x_i - x_j)**2 / (2*sigma_i**2))

print(p_ij)
