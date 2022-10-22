import numpy as np
import matplotlib.pyplot as plt

data1 = np.array([[0.8, 0.8, 0.6, 0.4, 0.2, 0.2, 0.4], [1.2, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8],
                 [0.6, 0.4, 0.4, 0.6, 0.6, 0.6, 1.0],[3.5, 3.2, 2.6, 2, 1.6, 1.6, 1.4]])

delta = data1[0]/data1[1]
print(delta)

V = (data1[3]- data1[2])/(data1[3] + data1[2])
V_1 = 2*np.sqrt(delta)/(1+delta)

V3 = V / V_1

cosa = np.cos(np.array([0, 15, 30, 45, 60, 75, 90])*np.pi/180)


print(V3)
print(abs(cosa))
a, b = np.polyfit(cosa, V3, deg=1)
X = np.linspace(0, 1, 1000)
Y = [a*x +b for x in X]
print(a)
plt.errorbar(abs(cosa), V3, yerr=0.15, fmt='s')
plt.plot(X, Y)
plt.xlabel('cos a', fontsize='17')
plt.ylabel('V3', fontsize='17')
plt.grid()
plt.show()

data2 = np.array([[0, 2, 4, 6, 8, 10, 13, 17, -2, -4, -6], [0.8, 0.1, 1.1, 1.1, 1, 1, 1.1, 2.8, 1.2, 1.2, 1.2],
                  [1.2, 1.1, 1.2, 0.8, 1.2, 0.4, 1.2, 2.2, 1.2, 1.2, 1.8], [0.6, 0.7, 0.9, 0.9, 1.6, 1, 2, 4.6,
                                                                            0.8, 0.8, 1.4],
                  [3.5, 1.6, 3.9, 2.8, 3.4, 2, 2.6, 5.2, 4, 3.8, 4.2]])

delta = data2[1]/data2[2]

V = (data2[4]- data2[3])/(data2[4] + data2[3])
V_1 = 2*np.sqrt(delta)/(1+delta)

V2 = V / V_1
print(V2)
plt.errorbar(data2[0], V2, fmt='s', yerr=0.05)
plt.xlabel('x, см', fontsize='17')
plt.ylabel('V2', fontsize='17')
plt.grid()
plt.show()
