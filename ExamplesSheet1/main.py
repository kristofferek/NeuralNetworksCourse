import matplotlib.pyplot as plt
import numpy as np

def f(x):
	return np.sqrt(x)

x1 = np.arange(0.0, 100.0, 1)

plt.plot(x1, f(x1), 'r.')
plt.show()