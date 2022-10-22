import numpy as np
import matplotlib.pyplot as plt

P = np.array([5565, 6600, 7913, 9330, 10570, 12170, 13350])
T = np.array([1233, 1288, 1311, 1355, 1465, 1476, 1498])

logP = np.log10(P)
logT = np.log10(T)

print(logP)
print(logT)

a, b = np.polyfit(logT, logP, deg=1)
X = np.linspace(logT[0], logT[-1], 1000)
Y = np.array([a*x + b for x in X])

plt.errorbar(logT, logP, fmt='s')
plt.plot(X, Y, label='logP = {} + {} logT'.format(round(b, 1), round(a, 2)))
plt.xlabel('log T', fontsize='17')
plt.ylabel('log P', fontsize='17')
plt.grid()
plt.legend(fontsize='14')
plt.show()

"""x = np.array([45.4, 45.8, 46.1, 46.3])
z = (47 - x)*0.1
z.sort()
n = np.array([2, 3, 4, 5])
y = 2*np.sqrt(z*n*546.1*10**-9)*1000
plt.errorbar(n, y, fmt='s', yerr=0.15)
plt.grid()
plt.xlabel('n', fontsize='17')
plt.ylabel(r'$2 z_n $, мм',fontsize='17')
plt.show()

x = [-1.4, -0.36, 0, 0.4, 1.6]
n = [-2, -1, 0, 1, 2]
a, b = np.polyfit(n, x, deg=1)
X = np.linspace(-2, 2, 1000)
Y = [a*x + b for x in X]
plt.errorbar(n, x, fmt='s', yerr=0.2)
plt.plot(X, Y)
plt.grid()
plt.xlabel('n', fontsize='17')
plt.ylabel(r'$x_n$, мм', fontsize='17')
plt.show()
print(a)"""