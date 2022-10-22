import matplotlib.pyplot as plt
import numpy as np
import statistics as st


def find_koef(x_list, y_list):
    n = len(x_list)
    x_av = 0
    x_sq_av = 0
    y_av = 0
    x_y_av = 0
    for i in range(n):
        x_av += x_list[i]
        x_sq_av += x_list[i]**2
        y_av += y_list[i]
        x_y_av += x_list[i]*y_list[i]

    x_av/=n
    y_av/= n
    x_sq_av /= n
    x_y_av /= n
    k = (x_y_av - x_av*y_av) / (x_sq_av - x_av**2)
    b = (y_av - k*x_av)

    return k, b

def find_angle_sigma_k(x, y, k):
    number_of_points = len(x)
    x_sqr_average = 0
    x_average = 0
    y_average = 0
    y_sqr_average = 0
    for i in range(number_of_points):
        x_sqr_average += x[i]**2
        y_sqr_average += y[i]**2
    x_sqr_average /= number_of_points
    y_sqr_average /= number_of_points

    sigma_k = np.sqrt(abs(y_sqr_average/x_sqr_average - k**2)) / np.sqrt(number_of_points)
    return sigma_k

V_1_list = [0.96, 1, 1.5, 1.5, 2, 2, 3, 3, 3, 4, 5, 4, 5, 5, 5, 5]
p_1_list = [7, 11, 20, 30, 40, 45, 55, 65, 75, 86, 95, 106, 113, 126, 145, 170]
t_1_list = [91.3, 70.2, 59.8, 40.4, 42.2, 38.4, 46.9, 40.0, 35.1, 41.3, 49.7, 38.0, 46.5, 45.85, 43.5, 41.7]
q_list = []

sigma_v = 0.01
sigma_t = st.stdev([40.0, 40.2, 40.15, 40.22, 40.2, 40.0, 40.2, 39.97, 40.17, 40.28, 40.20])
print(sigma_t)

sigma_p_list = []
sigma_q_list = []

n = len(p_1_list)
for i in range(n):
    p_1_list[i] = p_1_list[i]*9.8067*0.2*0.9910
    q_list.append(V_1_list[i]/t_1_list[i])
    sigma_p_list.append(0.5*9.8067*0.2*0.9910)
    sigma_q_list.append(q_list[i]*np.sqrt((sigma_t / t_1_list[i])**2 + (sigma_v / V_1_list[i])**2))

q_linear_list = [q_list[i] for i in range(10)]
p_linear_list = [p_1_list[i] for i in range(10)]

print(p_1_list)
print(q_list)

print(sigma_p_list)
print(sigma_q_list)

a, b = find_koef(q_linear_list, p_linear_list)
X = np.arange(q_list[0], q_list[-1], 0.003)
Y = [a*x + b for x in X]

sigma_a = find_angle_sigma_k(q_linear_list, p_linear_list, a)

eta = np.pi*(1.975*10**-3)**4 * a / (8*0.5) * 1000

sigma_eta = eta*np.sqrt((4*0.05/3.95)**2 + (sigma_a/a)**2)

print(eta, sigma_eta)

plt.errorbar(q_list, p_1_list, xerr=sigma_q_list, yerr=sigma_p_list, fmt='^', color='blue')
plt.plot(X, Y, color='blue')
plt.xlabel('Q, л/с', fontsize='17')
plt.ylabel(r'$\Delta P$, Па', fontsize='17')
plt.title(r'Зависимость $\Delta P (Q)$, $d = (3.95 \pm 0.05)$ мм', fontsize='15')
plt.grid()

plt.show()

dp_list = [25, 37, 52, 70]
dl_list = [10.9, 30, 40, 50]
x_list = [10.9, 40.9, 80.9, 130.9]

n1 = len(dp_list)
for i in range(n1):
    dp_list[i] = dp_list[i]*9.8067*0.2*0.9910

print(dp_list)

a1, b1 = find_koef(x_list, dp_list)
X1 = np.arange(x_list[0], x_list[-1], 0.003)
Y1 = [a1*x + b1 for x in X1]

plt.errorbar(x_list, dp_list, yerr=2, fmt='.', color='blue')
plt.plot(X1, Y1, color='blue')
plt.grid()
plt.xlabel('x, см', fontsize='17')
plt.ylabel('P, Па', fontsize='17')
plt.title(r'Зависимость P(x). $d = (3.95 \pm 0.05)$ мм', fontsize='15')
plt.show()






V_1_list = [0.9, 0.9, 1, 1, 1.5, 2.5, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
p_1_list = [5, 10 ,15, 20, 25, 30, 35, 40, 45, 50, 55, 65, 75, 90, 100, 111]
t_1_list = [71.9, 40.6, 29.3, 22.5, 26.2, 37.9, 37.8, 43.9, 39.5, 35.3, 33.4, 39.8, 37.2, 34.0, 32.8, 28.2]
q_list = []

sigma_v = 0.01
sigma_t = st.stdev([40.0, 40.2, 40.15, 40.22, 40.2, 40.0, 40.2, 39.97, 40.17, 40.28, 40.20])
print(sigma_t)

sigma_p_list = []
sigma_q_list = []

n = len(p_1_list)
for i in range(n):
    p_1_list[i] = p_1_list[i]*9.8067*0.2*0.9910
    q_list.append(V_1_list[i]/t_1_list[i])
    sigma_p_list.append(0.5*9.8067*0.2*0.9910)
    sigma_q_list.append(q_list[i]*np.sqrt((sigma_t / t_1_list[i])**2 + (sigma_v / V_1_list[i])**2))

q_linear_list = [q_list[i] for i in range(11)]
p_linear_list = [p_1_list[i] for i in range(11)]

print(p_1_list)
print(q_list)

print(sigma_p_list)
print(sigma_q_list)

a, b = find_koef(q_linear_list, p_linear_list)
X = np.arange(q_list[0], q_list[-1], 0.003)
Y = [a*x + b for x in X]

sigma_a = find_angle_sigma_k(q_linear_list, p_linear_list, a)

eta = np.pi*(2.55*10**-3)**4 * a / (8*0.8) * 1000

sigma_eta = eta*np.sqrt((4*0.05/5.1)**2 + (sigma_a/a)**2)

print(eta, sigma_eta)

plt.errorbar(q_list, p_1_list, xerr=sigma_q_list, yerr=sigma_p_list, fmt='^', color='blue')
plt.plot(X, Y, color='blue')
plt.xlabel('Q, л/с', fontsize='17')
plt.ylabel(r'$\Delta P$, Па', fontsize='17')
plt.title(r'Зависимость $\Delta P (Q)$, $d = (5.10 \pm 0.05)$ мм', fontsize='15')
plt.grid()

plt.show()


dp_list = [30, 54, 75, 103]
dl_list = [10.7, 30.7, 40.7, 50]
x_list = [10.7, 40.7, 80.7, 130.7]

n1 = len(dp_list)
for i in range(n1):
    dp_list[i] = dp_list[i]*9.8067*0.2*0.9910

print(dp_list)

a1, b1 = find_koef(x_list, dp_list)
X1 = np.arange(x_list[0], x_list[-1], 0.003)
Y1 = [a1*x + b1 for x in X1]

plt.errorbar(x_list, dp_list, yerr=2, fmt='s', color='blue')
plt.plot(X1, Y1, color='blue')
plt.grid()
plt.xlabel('x, см', fontsize='17')
plt.ylabel('P, Па', fontsize='17')
plt.title(r'Зависимость P(x). $d = (5.10 \pm 0.05)$ мм', fontsize='15')
plt.show()