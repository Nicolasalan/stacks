import numpy as np
import matplotlib.pyplot as plt

x = 5
total_time = 2
num_steps = 200
dt = total_time / num_steps

def f(x, t):
    # parte deterministica dx = f(x, τ)dτ + g(τ)dw -> drift
    return -1 * x

def g(t):
    # controla a intensidade do ruido
    return 1.5

trajec = [x]

for i in range(num_steps):
    dw = np.random.normal(loc=0.0, scale=np.sqrt(dt))
    dx = f(x, i*dt) * dt + g(i*dt)*dw
    x = x + dx
    trajec.append(x)

plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, total_time, num_steps+1), trajec)
plt.grid(True)
plt.show()
