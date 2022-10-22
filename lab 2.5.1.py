import numpy as np
import matplotlib.pyplot as plt

y = [79.0, 79.83, 80.60, 81.35, 82.86, 83.63, 84.39, 85.89]
x = [332, 328, 323, 318.1, 313.2, 308.3, 303.3, 297.4]
sigma_1 = 1.0*np.sqrt(2)
sigma_2 = 0.01
d = 1.55
dp_list = [204.0, 206.0, 208.0, 209.9, 213.9, 215.8, 217.8, 221.7]
sigma_y = []
for i in range(len(dp_list)):
    sigma_y.append(y[i]*np.sqrt((sigma_1/dp_list[i])**2 + (sigma_2/d)**2))
print(sigma_y)

def find_k_chi_sqr(data_list_x, data_list_y, data_list_sigma):
    a = b = c = d = f = 0
    n = len(data_list_x)
    for i in range(n):
        a += data_list_x[i]**2 / data_list_sigma[i]**2
        b += data_list_y[i] / data_list_sigma[i]**2
        c += data_list_x[i] / data_list_sigma[i]**2
        d += 1 / data_list_sigma[i]**2
        f += data_list_x[i]*data_list_y[i] / data_list_sigma[i]**2
    k = (f*d - b*c) / (d*a - c**2)
    m = (b - k*c) / d
    return k, m

def find_angle_sigma_k(x, y, k):
    number_of_points = len(x)
    x_sqr_average = 0
    x_average = 0
    y_average = 0
    y_sqr_average = 0
    for i in range(number_of_points):
        x_sqr_average += x[i]**2
        y_sqr_average += y[i]**2
        x_average += x[i]
        y_average += y[i]
    x_average /= number_of_points
    y_average /= number_of_points
    x_sqr_average /= number_of_points
    y_sqr_average /= number_of_points

    sigma_k = (np.sqrt((y_sqr_average - y_average**2 )/ (x_sqr_average - x_average**2) - k**2)) / np.sqrt(number_of_points - 1)
    return sigma_k


k, b = find_k_chi_sqr(x, y, sigma_y)
sigma_k = find_angle_sigma_k(x, y, k)
print(k, sigma_k)

dsigma_dt_list = [(y[i] - y[i - 1]) / (x[i] - x[i-1]) for i in range(1, len(x))]


plt.errorbar(x, y, xerr=0.1, yerr=sigma_y, fmt='s', color='blue')
X = np.arange(297.4, 332, 0.003)
Y = [k*a + b for a in X]
plt.grid()
plt.plot(X, Y,color='blue')
plt.ylabel(r'$\sigma$, мН/м')
plt.xlabel('T, К')
plt.title(r'Зависимость $\sigma (T)$')
plt.show()

y_1 = [-T*k*10**(-3) for T in x]
y_2 = [(y[i] - x[i]*k)*10**(-3) for i in range(len(x))]


sigma_y_1 = [np.sqrt((sigma_k / k)**2 + (0.1 / x[i])**2)*y_1[i]*10**-3 for i in range(len(x))]
sigma_y_2 = [np.sqrt(sigma_y_1[i]**2 + (sigma_y[i]*10**-3)**2) for i in range(len(x))]
k_1, b_1 = find_k_chi_sqr(x, y_1, sigma_y_1)
k_2, b_2 = find_k_chi_sqr(x, y_2, sigma_y_2)
plt.errorbar(x, y_1, xerr=0.1, yerr=sigma_y_1, fmt='s', color='blue', label=r'$-T\frac{d \sigma}{d T}$')
plt.errorbar(x, y_2, xerr=0.1, yerr=sigma_y_2, fmt='s', color='red', label=r'$\sigma - T\frac{d \sigma}{d T}$')
Y_1 = [k_1*a + b_1 for a in X]
Y_2 = [k_2*a + b_2 for a in X]
plt.plot(X, Y_1, color='blue')
plt.plot(X, Y_2, color='red')
plt.grid()
plt.legend(fontsize='20')
plt.xlabel('T, К', fontsize='20')
plt.ylabel(r'q, u, Дж/м$^2$', fontsize='20')
plt.show()

y_1 = [y_1[i]*1000 for i in range(len(y_1))]
sigma_y_1 = [sigma_y_1[i]*1000 for i in range(len(y_1))]

y_2 = [y_2[i]*1000 for i in range(len(y_1))]
sigma_y_2 = [sigma_y_2[i]*1000 for i in range(len(y_1))]


dsigma_dt_list = [-0.166, -0.154, -0.15, -0.154, -0.152]
x_1_1 = [330, 326, 320, 310, 305]
y_1_1 = [-x_1_1[i]*dsigma_dt_list[i] for i in range(len(x_1_1))]

a1, b1 = np.polyfit(x_1_1, y_1_1, deg=1)
print(a1, b1)

X = np.arange(x_1_1[-1], x_1_1[0], 0.003)
Y = [a1*x + b1 for x in X]

plt.errorbar(x_1_1, y_1_1, yerr=[3, 1.8, 2, 1, 1] ,fmt='s', color='blue')
plt.plot(X, Y, color='blue')
plt.grid()
plt.xlabel('T, K', fontsize='17')
plt.ylabel('q, мДж / м^2', fontsize='17')
plt.show()

print(y_1_1)

y_1_2 = [79.415 + y_1_1[0], 80.215 + y_1_1[1], 80.975 + y_1_1[2], 83.245 + y_1_1[3], 84.01 + y_1_1[4]]

plt.errorbar(x_1_1, y_1_2, fmt='s')
plt.grid()
plt.xlabel('T, K')
plt.ylabel('U, мДж/м^2')
plt.show()
