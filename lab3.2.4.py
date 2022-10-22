import numpy as np
import matplotlib.pyplot as plt

"""C_list = [0.02, 0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82, 0.92]
x_list = [10, 5, 4.6, 4, 3, 3.6, 3.8, 11.6, 10, 11]
x_0_list = [10, 10, 10, 10, 10, 10, 10, 12.8, 11, 13]
n_list = [7, 6, 4, 3, 2, 2, 2, 6, 5 ,5]
L = 134.5 *10**(-3)
T_theor = [2*np.pi*np.sqrt(c*10**(-6)*L)*1000 for c in C_list]
T_real = [0.01*x_list[i]/(n_list[i]*x_0_list[i])*1000 for i in range(len(x_list))]

plt.errorbar(T_theor, T_real, yerr=0.15, fmt='s', color='blue')
plt.xlabel(r'$T_{теор}$, мс',fontsize='17')
plt.ylabel(r'$T_{эксп}$, мс', fontsize='17')
plt.grid()
plt.show()
print(T_real)
print(T_theor)"""

n_list = [3, 3, 3, 2, 2, 2, 2]
U_K_list = [5.2, 4.6, 6.8, 4.6, 3, 4.2, 7.8]
U_K_n_list = [0.8, 0.4, 0.4, 0.4, 0.2, 0.4, 0.2]
R_list = [910, 1210, 1510, 1810, 2110, 2410, 2710]
theta_list = [1/n_list[i]*np.log(U_K_list[i]/U_K_n_list[i]) for i in range(len(n_list))]
y_list = [1/theta**2 for theta in theta_list]
x_list = [1/r**2 *1000000 for r in R_list]
a, b = np.polyfit(x_list, y_list, deg=1)
print(a)
X = np.arange(x_list[-1], x_list[0], 0.003)
Y = [a*x + b for x in X]

plt.errorbar(x_list, y_list, yerr=0.2 , fmt='s', color='red')
plt.plot(X, Y, color='blue')
plt.xlabel(r'$\frac{1}{R^2}$, Ом$^{-2}$ * $10^{-6}$', fontsize='17')
plt.ylabel(r'$\frac{1}{\Theta^2}$', fontsize='17')
plt.grid()
plt.show()
