from numpy import require
import torch
import pandas as pd
x = torch.tensor(1.0, requires_grad=True)

y = x**2
y.backward()
# x.grad.zero_()
y2 = x**3
y2.backward()
print("Valor de y com backward: ", x.grad)

# modelo linear simples

def normalize(df):
    return (df - df.min()) / (df.max() - df.min())

df = pd.DataFrame(
    {
        "area": [120, 130, 140, 150],
        "idade": [2, 3, 4, 5],
        "preco": [30, 40, 50, 60]
    }
)

df = normalize(df)

print(df.to_string(index=False))

X = torch.tensor(df[["area", "idade"]].values, dtype=torch.float32)
Y = torch.tensor(df[["preco"]].values, dtype=torch.float32)

print("X: ", X, "Y: ", Y)

W = torch.rand(size=(2, 1), requires_grad=True)
B = torch.rand(1, requires_grad=True)

pred = X @ W + B
error = (Y-pred)**2
loss = error.mean()
print(f"Loss: {loss.item()}, Error: {error}, pred: {pred}")

print()
loss.backward()
dW = W.grad
dB = B.grad
print("dW: ", dW, "dB: ", dB)

lr = 0.2
with torch.no_grad():
    W = W - lr * dW
    B = B - lr * dB

print(f"Novos pesos -> w: {W}, b: {B}")

pred = X @ W + B
loss = ((Y-pred)**2).mean()
print("Novo loss: ", loss.item())
