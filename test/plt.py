import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 8, 1000)
y1 = np.sin(x)
y2 = np.arcsin(x)

plt.plot(x, y1, label='sine')
plt.plot(x, y2, label='arcsine')
plt.legend()
plt.show()