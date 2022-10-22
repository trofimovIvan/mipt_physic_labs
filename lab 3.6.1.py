import numpy as np
import matplotlib.pyplot as plt


d_v_list = [5, 5.5, 7, 10, 11, 11, 40]
tau_obr_list = [5.6, 6.7, 8.0, 10.0, 13.3, 20.0, 40.0]
sigma_d_v_list = [0.9, 1.1, 2.1, 3.1, 4.0, 4.0, 4.0]
sigma_tau_obr = 0.5

a, b = np.polyfit(tau_obr_list, d_v_list, deg=1)
X = np.arange(tau_obr_list[0], tau_obr_list[-1], 0.001)
Y = [a*x + b for x in X]
print(a, b)

plt.errorbar(tau_obr_list, d_v_list, xerr=sigma_tau_obr, yerr=sigma_d_v_list, fmt='.', color='red')
plt.plot(X, Y, color='blue')
plt.xlabel(r'$\frac{1}{\tau}$, $10^{-3}$ мкс$^{-1}$', fontsize='17')
plt.ylabel(r'$\Delta \nu $, кГц', fontsize='17')
plt.grid()
plt.show()

d_v_list = [1, 1.8, 3.0, 8.0, 10.0]
f_rep_list = [1, 2, 3, 5, 7.5]
sigma_d_v_list = [0.1, 0.4, 0.4, 1.0, 1.0]
a, b = np.polyfit(f_rep_list, d_v_list, deg=1)
X = np.arange(f_rep_list[0], f_rep_list[-1], 0.001)
Y = [a*x + b for x in X]
plt.errorbar(f_rep_list, d_v_list, yerr=sigma_d_v_list, fmt='.', color='blue')
plt.plot(X, Y, color='red')
plt.grid()
plt.xlabel(r'$f_{повт}$, кГц', fontsize='17')
plt.ylabel(r'$\Delta \nu$, кГц', fontsize='17')
print(a, b)
plt.show()



