import numpy as np
import matplotlib.pyplot as plt

def read_file(file_name, delta_t):
    data_list = []
    time_list = []
    u_u_0_list = []
    time = 0
    with open('{}'.format(file_name), 'r') as my_data_file:
        first_value = float(my_data_file.readline())
    with open('{}'.format(file_name), 'r') as my_data_file:
        for string in my_data_file:
            data_list.append(-np.log(float(string) / first_value))
            u_u_0_list.append(float(string)/ first_value)
            time_list.append(time)
            time += delta_t
    return data_list, time_list, u_u_0_list

def find_koef(x_list, y_list):
    n = len(x_list)
    znam = 0
    chisl = 0
    for i in range(n):
        chisl += x_list[i]*y_list[i]
        znam += x_list[i]**2
    return chisl / znam

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
    return k, b

def get_plot(x_list1, x_list2, x_list3, x_list4, y_list1, y_list2, y_list3, y_list4):

    k1 = find_koef(x_list1, y_list1)
    sigma_k1 = find_angle_sigma_k(x_list1, y_list1, k1)
    print(sigma_k1, k1)
    D1 = 800* 15 * k1 / 2
    sigma_D1 = D1 * np.sqrt(( 5 / 800)**2 + (sigma_k1/k1)**2 + (0.1/15)**2)
    plt.errorbar(x_list1, y_list1, fmt='r^', label=r'$(45 \pm 1.6) $ торр ')
    X1 = np.arange(x_list1[0], x_list1[-1], 0.003)
    Y1 = [k1*x for x in X1]
    plt.plot(X1, Y1, color='red')

    k2 = find_koef(x_list2, y_list2)
    sigma_k2 = find_angle_sigma_k(x_list2, y_list2, k2)
    D2 = 800 * 15 *k2 / 2
    sigma_D2 = D2 * np.sqrt((5 / 800) ** 2 + (sigma_k2 / k2) ** 2 + (0.1 / 15) ** 2)
    plt.errorbar(x_list2, y_list2, fmt='b^', label=r'$(261 \pm 1.6) $ торр ')
    X2 = np.arange(x_list2[0], x_list2[-1], 0.003)
    Y2 = [k2 * x for x in X2]
    plt.plot(X2, Y2, color='blue')

    k3 = find_koef(x_list3, y_list3)
    sigma_k3 = find_angle_sigma_k(x_list3, y_list3, k3)
    D3 = 800 * 15 * k3 / 2
    sigma_D3 = D3 * np.sqrt((5 / 800) ** 2 + (sigma_k3 / k3) ** 2 + (0.1 / 15) ** 2)
    plt.errorbar(x_list3, y_list3, fmt='g^', label=r'$(127 \pm 1.6)$ торр ')
    X3 = np.arange(x_list3[0], x_list3[-1], 0.003)
    Y3 = [k3 * x for x in X3]
    plt.plot(X3, Y3, color='green')

    k4 = find_koef(x_list4, y_list4)
    sigma_k4 = find_angle_sigma_k(x_list4, y_list4, k4)
    D4 = 800 * 15 * k4 / 2
    sigma_D4 = D4 * np.sqrt((5 / 800) ** 2 + (sigma_k4 / k4) ** 2 + (0.1 / 15) ** 2)
    plt.errorbar(x_list4, y_list4, fmt='k^', label=r'$(395 \pm 1.6)$ торр ')
    X4 = np.arange(x_list4[0], x_list4[-1], 0.003)
    Y4 = [k4 * x for x in X4]
    plt.plot(X4, Y4, color='black')

    plt.xlabel('t, с', size='20')
    plt.ylabel(r'-ln $\frac{U}{U_0}$', size='20')
    plt.legend()
    plt.title(r'Зависимость $-ln\frac{U}{U_0}$ от t')

    plt.grid()
    plt.show()
    return [D1, D2, D3, D4], [sigma_D1, sigma_D2, sigma_D3, sigma_D4]


def get_plot_final(P_rev_list, D_list, sigma_P_rev_list, sigma_D_list):
    plt.errorbar(P_rev_list, D_list,  xerr=sigma_P_rev_list, yerr=sigma_D_list, fmt='b^')
    k = find_koef(P_rev_list, D_list)
    print(find_angle_sigma_k(P_rev_list, D_list, k))
    k, b = np.polyfit(P_rev_list, D_list, deg=1)
    sigma_k = find_angle_sigma_k(P_rev_list, D_list, k)

    X = np.arange(P_rev_list[0], P_rev_list[-1], 0.5)
    Y = [k*x + b for x in X]
    print(k*10000/760)
    print((k+sigma_k)*10000/760)
    print((k-sigma_k)*10000/760)
    plt.plot(X, Y, color='blue')
    plt.xlabel(r'$\frac{1}{p}$, $торр^{-1} * 10^{-4}$', size='20')
    plt.ylabel(r'D $\frac{см^2}{c}$', size='20')
    plt.title(r'Зависимость D от $\frac{1}{P}$')
    plt.grid()
    plt.show()

data_list_1, time_list_1, pas1 = read_file('data 2.2.1 p1 = 45 torr', 10)
data_list_2, time_list_2, pas2 = read_file('data 2.2.1 p2 = 261 torr', 15)
data_list_3, time_list_3, pas3 = read_file('data 2.2.1 p3 = 127 torr', 10)
data_list_4, time_list_4, pas4 = read_file('data 2.2.1 p4 = 395 torr', 30)

D_list, sigma_D_list = get_plot(time_list_1, time_list_2, time_list_3, time_list_4,
         data_list_1, data_list_2, data_list_3, data_list_4)

print(D_list)
print(sigma_D_list)

D_list.sort()
P_rev_list = [1/45 * 10000, 1/127 * 10000, 1/261 * 10000, 1/395* 10000]
P_rev_list.sort()
print(P_rev_list)
print(D_list)
get_plot_final(P_rev_list, D_list, [0.1, 0.23, 0.99, 7.90], [0.04, 0.026, 0.04, 0.1])



