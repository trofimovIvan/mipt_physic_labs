import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def f(x_data, A, B, C):
    return A*np.tanh(B*x_data) + C*x_data

"""V_1_list_straight = [25.49, 25.06, 24.59, 24.10, 24.06, 23.9, 23.74, 23.62]
I_1_list_straight = [2.16, 2.40, 2.80, 3.20, 3.60, 4.00, 4.40, 4.80]
V_1_list_back = [23.77, 23.91, 24.05, 24.07, 24.54, 25.05, 25.81, 26.98, 29.77, 34.24, 35.50]
I_1_list_back = [4.40, 4.00, 3.60, 3.20, 2.80, 2.40, 2.00, 1.60, 1.20, 0.80, 0.40]

X = [1.60, 1.20, 0.80]
Y = [26.98, 29.77, 34.24]
a,b = np.polyfit(X, Y, deg=1)
X_1 =np.arange(0.5, 1.9, 0.001)
Y_1 = [a*x + b for x in X_1]


plt.errorbar(I_1_list_straight, V_1_list_straight, xerr=0.05, yerr=0.01,
             color='red', label='Увеличение тока', fmt='s')
plt.errorbar(I_1_list_back, V_1_list_back, xerr=0.05, yerr=0.01,
             color='blue', label='Уменьшение тока', fmt='s')
plt.plot(X_1, Y_1, color='green')
plt.ylabel('V, В', fontsize='20')
plt.xlabel('I, мА', fontsize='20')
plt.grid()
plt.legend(fontsize='13')
plt.show()
print(a)"""

"""V_2_list_1 = [-25.06, -22.02, -19.06, -16.02, -13.07, -10.10, -8.03, -6.05, -4.02, -2.09,
            -1.0, 0.53, 1.09, 2.05, 4.00, 6.00, 7.99, 10.02, 13.05, 16.07, 19.08, 22.05, 25.03]
I_2_list_1 = [-111.37, -110.75, -109.00, -105.30, -98.00, -84.40, -71.2, -54.9, -34.4, -11.7, -1.09,
              5.3, 12.1, 24.0, 46.6, 66.56, 82.8, 95.8, 109.4, 116.9, 121.0, 122.6, 120.0]"""

V_2_list_1 = [-25.06, -22.04, -19.03, -16.08, -13.03, -10.08, -8.05, -6.06, -4.07, -2.01, -0.93, 0.57,
              2.02, 4.07, 6.00, 8.04, 10.00, 13.00, 16.08, 19.10, 22.04, 25.06]
I_2_list_1 = [-68.51, -66.35, -64.32, -62.04, -58.34, -51.84, -44.95, -35.72, -23.80, -9.14, -1.04, 6.95,
              18.07, 32.44, 43.95, 53.45, 60.30, 67.11, 71.06, 73.62, 75.81, 77.97]

"""V_2_list_1 = [-25.06, -22.07, -19.05, -16.07, -13.04, -10.06, -8.05, -6.06, -4.07, -2.07, -0.86, 0.46,
              0.95, 2.03, 4.08, 6.04, 8.06, 10.09, 13.07, 16.06, 19.08, 22.02, 25.06]
I_2_list_1 = [-33.45, -32.29, -31.18, -30.02, -28.53, -25.86, -22.88, -18.61, -12.01, -5.78, -0.95, 3.84, 5.86,
              10.15, 17.60, 23.38, 27.85, 31.03, 33.94, 35.68, 37.08, 38.39, 39.83]"""

popt = curve_fit(f, V_2_list_1, I_2_list_1)[0]
print(popt)
"""X = np.arange(V_2_list_1[0], V_2_list_1[-1], 0.001)
Y = [f(x, popt[0], popt[1], popt[2]) for x in X]

plt.errorbar(V_2_list_1, I_2_list_1, xerr=0.01, yerr=0.05, fmt='s', color='blue')
plt.plot(X, Y, color='red')
plt.xlabel('V, В', fontsize='20')
plt.ylabel('I, мА', fontsize='20')
plt.grid()
plt.show()"""

e = 1.6*10**-19
mi = 22*1.66*10**-24
k = 1.38*10**-23
m_e = 9.1 * 10**(-28)
In = popt[0]*10**-6
print(In)
didu = (In*popt[1] + popt[2]*10**-6)
T_e = In/(2*didu)
print(T_e)
T_e *= 11800

n_e = In * np.sqrt(mi) / (0.4*e*np.pi*0.02*0.52*np.sqrt(2*k*T_e))
print(n_e / 10**6)
w_p = 5.6*10**4 * np.sqrt(n_e / 10**6)
print(w_p)
rde = np.sqrt(k*T_e/(4*np.pi*n_e*e**2))
print(rde)
rd = np.sqrt(k*300/(4*np.pi*n_e*e**2))
N_d = 4/3 * np.pi * n_e*rd**3
print(rd)
print(N_d)
alpha = 2/760 * 300 / T_e
print(alpha)
