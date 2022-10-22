import numpy as np
import matplotlib.pyplot as plt

delta = 4

data1 = delta* np.array([-107, -70, -32, 0, 41, 83, 122])
data2 = delta* np.array([-208, -100, 0, 101, 198])
data3 = delta* np.array([-67, -32, 0, 29, 65])
m1 = np.array([-3, -2, -1, 0, 1, 2, 3])
m2 = np.array([-2, -1, 0, 1, 2])
m3 = np.array([-2, -1, 0, 1, 2])

a1, b1 = np.polyfit(m1, data1, deg=1)
X1 = np.arange(-3, 3, 0.003)
Y1 = [a1*x + b1 for x in X1]

a2, b2 = np.polyfit(m2, data2, deg=1)
X2 = np.arange(-2, 2, 0.003)
Y2 = [a2*x + b2 for x in X2]

a3, b3 = np.polyfit(m3, data3, deg=1)
X3 = np.arange(-2, 2, 0.003)
Y3 = [a3*x + b3 for x in X3]

plt.errorbar(m1, data1, color='blue', fmt='s', label=r'$\nu = 1.18 МГц$')
plt.errorbar(m2, data2, color='red', fmt='s', label=r'$\nu = 2.9 МГц$')
plt.errorbar(m3, data3, color='green', fmt='s', label=r'$\nu = 1.0 МГц$')

plt.plot(X1, Y1, color='blue')
plt.plot(X2, Y2, color='red')
plt.plot(X3, Y3, color='green')

plt.xlabel('m', fontsize='17')
plt.ylabel(r'$x_m$, мкм', fontsize='17')
plt.legend(fontsize='14')
plt.grid()
plt.show()

print('lm_1 = ', a1)
print('lm_2 = ', a2)
print('lm_3 = ', a3)
print('L1 = ', 625*10**-9 * 0.3/a1 * 10**6 * 10**3)
print('L1 = ', 625*10**-9 * 0.3/a2 * 10**6 * 10**3)
print('L1 = ', 625*10**-9 * 0.3/a3 * 10**6 * 10**3)
print('v1 = ', 625*10**-9 * 0.3/a1 * 10**6 * 1.18*10**6)
print('v2 = ', 625*10**-9 * 0.3/a2 * 10**6 * 2.9*10**6)
print('v3 = ', 625*10**-9 * 0.3/a3 * 10**6 * 10**6)

C = 0.8
N = np.array([9.3, 9.9, 9.8, 4])
n = np.array([10, 9, 8, 10])
L = C*N/(n - 1)
nu = np.array([1.82, 1.54, 1.18, 4.1])*10**6
print(L)
v = nu*L*10**-3
print(v)