import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:

    # if matrix.dim() == 1:
    #     m = torch.zeros_like(matrix)
    #     matrix = torch.stack([matrix, m])

    eigenvalues = torch.linalg.eig(matrix).eigenvalues
    sort, _ = torch.sort(eigenvalues.real)
    return sort


a = [[2.0, 1.0], [1.0, 2.0]]
b = [[0.0,1.0],[1.0,0.0]]
c = [[4.0,2.0],[1.0,3.0]]

a_t = torch.as_tensor(a, dtype=torch.float32)
b_t = torch.tensor(b, dtype=torch.float32)
c_t = torch.tensor(c, dtype=torch.float32)

print(calculate_eigenvalues(a_t))
print(calculate_eigenvalues(b_t))
print(calculate_eigenvalues(c_t))
