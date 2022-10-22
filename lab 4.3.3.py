import numpy as np
import matplotlib.pyplot as plt

d = np.array([58.8, 122.8, 151.0])
D = np.array([1.8, 0.92, 0.71])
D_new = 1/D
a, b = np.polyfit(D_new, d, deg=1)
X = np.arange(D_new[0], D_new[-1], 0.003)
Y = [a*x + b for x in X]
print(a)
plt.errorbar(D_new, d, fmt='s')
plt.xlabel(r'$\frac{1}{D}$, мм$^{-1}$', fontsize='17')
plt.ylabel(r'd, мкм',fontsize='17')
plt.plot(X, Y)
plt.grid()
plt.show()
