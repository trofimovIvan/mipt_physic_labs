import numpy as np
import matplotlib.pyplot as plt

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

    sigma_k = (np.sqrt((y_sqr_average - y_average**2 )/ (x_sqr_average - x_average**2) - k**2)) / np.sqrt(number_of_points)
    return sigma_k


def get_data(name, log) :
    file_data = open('{}'.format(name), 'r')

    temprature_list = []
    h_down_list = []
    h_up_list = []
    result_list_x = []
    result_list_y = []

    for line in file_data:
        temprature_list.append(float(line.split()[0]))
        h_down_list.append(float(line.split()[1]))
        h_up_list.append(float(line.split()[2]))

    file_data.close()

    if log:
        for i in range(len(h_down_list)):
            result_list_x.append(1 / (273 + temprature_list[i])* 10**3)
            result_list_y.append(np.log((h_up_list[i] - h_down_list[i])))
    else:
        for i in range(len(h_down_list)):
            result_list_x.append(273 + temprature_list[i])
            result_list_y.append((h_up_list[i] - h_down_list[i]))

    return result_list_x, result_list_y

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
    return k

result_list_x, result_list_y = get_data('data 2.4.1', True)
result_list_x_back, result_list_y_back = get_data('data 2.4.1_back', True)



plt.grid()

plt.scatter(result_list_x, result_list_y,s=100, c='blue', marker="_")
plt.scatter(result_list_x, result_list_y, s=300,c='blue',  marker="|")

plt.scatter(result_list_x_back, result_list_y_back,s=100, c='orange', marker="_")
plt.scatter(result_list_x_back, result_list_y_back, s=300,c='orange',  marker="|")

a, b = np.polyfit(result_list_x, result_list_y, deg=1)
a1, b1 = np.polyfit(result_list_x_back, result_list_y_back, deg=1)
print(-8.31*a, -8.31*a1)
print(find_angle_sigma_k(result_list_x, result_list_y, a)*8.31,
      find_angle_sigma_k(result_list_x_back, result_list_y_back, a1)*8.31)

X = np.arange(3.19, 3.39, 0.003)
true_list_y = [a*x + b for x in X]
true_list_y_back = [a1*x + b1 for x in X]
plt.plot(X, true_list_y, label='нагрев')
plt.plot(X, true_list_y_back, label='охлаждение')
plt.legend()
plt.title(r'Зависимость $ln(\frac{P}{P_0})$ от $\frac{1}{T}$')
plt.xlabel(r'$\frac{1}{T}$ $K^{-1} * 10^{-3}$')
plt.ylabel(r'$ln(\frac{P}{P_0})$')
plt.show()

result_list_x, result_list_y = get_data('data 2.4.1', False)
result_list_x_back, result_list_y_back = get_data('data 2.4.1_back', False)

plt.grid()
plt.errorbar(result_list_x, result_list_y, xerr=0.15, yerr=0.8, label='нагрев')
plt.errorbar(result_list_x_back, result_list_y_back, xerr=0.15, yerr=0.7, label='охлаждение')
plt.legend()
plt.title(r'Зависимость P от T')
plt.xlabel('T, K')
plt.ylabel('P, мм.рт.ст.')
plt.show()

res_list = []
for i in range(1, len(result_list_x)):
    res_list.append((-result_list_y[i - 1] + result_list_y[i]) / (result_list_x[i - 1] + result_list_x[i]) * 8.31 *
                    ((result_list_x[i] + result_list_x[i - 1]) / 2 )**2 / (result_list_y[i] + result_list_y[i-1]))

print(len(res_list))
print(res_list)
average = 0
for res in res_list:
    average += res
n = len(res_list)
average /= n
sigma = 0
for i in range(n):
    sigma += (res_list[i] - average)**2
sigma = sigma / (n*(n-1))
sigma = np.sqrt(sigma)
print(average)
print(sigma)



res_list = []
for i in range(1, len(result_list_x_back)):
    res_list.append((-result_list_y_back[i - 1] + result_list_y_back[i]) /
                    (result_list_x_back[i - 1] + result_list_x_back[i]) * 8.31 *
                    ((result_list_x_back[i] + result_list_x_back[i - 1]) / 2 )**2 /
                    (result_list_y_back[i] + result_list_y_back[i-1]))

print(len(res_list))
print(res_list)
average = 0
for res in res_list:
    average += res
n = len(res_list)
average /= n
sigma = 0
for i in range(n):
    sigma += (res_list[i] - average)**2
sigma = sigma / (n*(n-1))
sigma = np.sqrt(sigma)
print(average)
print(sigma)

