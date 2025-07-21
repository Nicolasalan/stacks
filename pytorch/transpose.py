import torch

def transpose_matrix(a) -> torch.Tensor:
    """
    Transpose a 2D matrix `a` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a transposed tensor.
    """
    a_t = torch.as_tensor(a)
    return torch.transpose(a_t, 0, 1)

a = [[1,2,3],[4,5,6]]
print(transpose_matrix(a))
