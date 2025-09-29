import torch

def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of means or raises ValueError on invalid mode.
    """
    a_t = torch.as_tensor(matrix, dtype=torch.float32)

    rows, cols = a_t.shape

    mean = []

    if mode == "column":
        for j in range(cols):
            x = torch.mean(a_t[:, j])
            mean.append(x.item())
    else:
        for i in range(rows):
            x = torch.mean(a_t[i, :])
            mean.append(x.item())

    return torch.tensor(mean, dtype=torch.float32)


matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
mode = "column"


print(calculate_matrix_mean(matrix, mode))
