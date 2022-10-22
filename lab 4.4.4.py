import numpy as np
import matplotlib.pyplot as plt

"""x1 = np.array([157.909, 154.077, 151.491, 149.377, 147.503, 145.915])
x2 = np.array([166.015, 171.466, 174.157, 176.353, 178.159, 179.812])
d = x2 - x1
dsqr = d**2
x = range(1, len(dsqr)+1)
a, b = np.polyfit(x, dsqr, deg=1)
X = np.arange(x[0], x[-1], 0.003)
Y = np.array([a*i + b for i in X])
print(dsqr)
plt.errorbar(range(1, len(dsqr)+1), dsqr, fmt='s')
plt.plot(X, Y)
plt.xlabel('N', fontsize='17')
plt.ylabel(r'$d_N^2, мм^2$', fontsize='17')
plt.grid()
plt.show()

print(a)"""

x1 = np.array([155.158, 152.303, 150.073, 148.102, 146.405])
x2 = np.array([170.335, 173.376, 175.626, 177.609, 179.318])
x11 = np.array([154.077, 151.491, 149.377, 147.503, 145.915])
x22 = np.array([171.466, 174.157, 176.353, 178.159, 179.812])

d1 = x2 - x1
d2 = x22 - x11
d_av = (d1+d2)/2
dd = 1/np.abs(d2 - d1)
print(dd)


a, b = np.polyfit(dd, d_av, deg=1)
X = np.arange(dd[0], dd[-1], 0.003)
Y = np.array([a*i + b for i in X])
print(a)
print(d_av)
plt.errorbar(dd, d_av, fmt='s')
plt.plot(X, Y)
plt.xlabel(r'$\frac{1}{\Delta d}, мм^{-1}$', fontsize='17')
plt.ylabel(r'$\overline{d}, мм$', fontsize='17')
plt.grid()
plt.show()