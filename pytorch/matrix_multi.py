import torch

def matrix_dot_vector(a: list, b: list) -> torch.Tensor:
    """
    Compute the product of matrix `a` and vector `b` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of length m, or tensor(-1) if dimensions mismatch.
    """
    a_t = torch.as_tensor(a, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)

    print("Size a: ", a_t.size(1))
    print("Size b: ", b_t.size(0))

    # Dimension mismatch check
    if a_t.size(1) != b_t.size(0):
        return torch.tensor(-1)
    else:
        matrix = torch.matmul(a_t, b_t)
        return matrix


a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
b = [1.0, 2.0]

print(matrix_dot_vector(a, b))
# output [5, 10]
