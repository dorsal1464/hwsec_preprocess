from matplotlib import pyplot as plt
from matplotlib import cm, ticker
import numpy as np

fig, ax = plt.subplots()

y = np.arange(0, 10, 0.6)
x = [0] * len(y)
z = np.random.randn(len(y))

cs = ax.scatter(x, y, c=z, cmap=cm.jet)

y = np.arange(0, 10, 0.3)
x = [1] * len(y)
z = np.random.randn(len(y))

cs = ax.scatter(x, y, c=z, cmap=cm.jet)

y = np.arange(0, 10, 0.9)
x = [2] * len(y)
z = np.random.randn(len(y))

cs = ax.scatter(x, y, c=z, cmap=cm.jet)

cbar = fig.colorbar(cs)

plt.show()