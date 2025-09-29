import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# preparando dados
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
print(X.shape, y.shape)
print(X.shape, y.shape[0])
y = y.view(y.shape[0], 1) # seria duas colunas, mas so tem 100 valores, nao teria como inventar mais 100 valores
print("y feito alguma coisa: ", X.shape, y.shape)

n_samples, n_features = X.shape

input_size = n_features
output_size = 1

# Criar modelo
model = nn.Linear(input_size, output_size)

learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 1000

for epochs in range(num_epochs):
    # passada para frente ou foward
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # passada para traz ou backward
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if (epochs+1) % 10 == 0:
        print(f'epoch: {epochs+1}, loss = {loss.item()}:.4f')

predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
