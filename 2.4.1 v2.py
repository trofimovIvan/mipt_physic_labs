import numpy as np
import matplotlib.pyplot as plt

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
            result_list_y.append(133.3*(h_up_list[i] - h_down_list[i]))

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
    m = (b - k*c)/d
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

    sigma_k = (np.sqrt((y_sqr_average - y_average**2 )/ (x_sqr_average - x_average**2) - k**2)) / \
              np.sqrt(number_of_points - 2)
    return sigma_k

def get_norm_const(data_sigma):
    w = 0
    for i in range(len(data_sigma)):
        w += 1 / data_sigma[i]**2

    return w

result_list_x, result_list_y = get_data('data 2.4.1', True)
result_list_x_back, result_list_y_back = get_data('data 2.4.1_back', True)

data_sigma_y = [0.0073, 0.0063, 0.0061, 0.0058, 0.0055, 0.0051, 0.0048, 0.0045, 0.00432, 0.004, 0.0038, 0.0035,
                0.0033, 0.0032]

data_sigma_y_back = [0.0073, 0.0069, 0.006, 0.0058, 0.0055, 0.0053, 0.005, 0.0046, 0.0043, 0.0042, 0.004,
                     0.0037, 0.0035, 0.0033]

w = get_norm_const(data_sigma_y)

k, m = find_k_chi_sqr(result_list_x, result_list_y, data_sigma_y)

print(-8.31*k)

k_back, m_back = find_k_chi_sqr(result_list_x_back, result_list_y_back, data_sigma_y_back)
print(-8.31*k_back)

X = np.arange(3.22, 3.39, 0.003)
Y = [k*x + m for x in X]
Y_back = [k_back*x + m_back for x in X]

plt.errorbar(result_list_x, result_list_y, yerr=data_sigma_y, xerr=1.1*10**-3, fmt='b^', label="Нагрев")
plt.errorbar(result_list_x_back, result_list_y_back, yerr=data_sigma_y_back, xerr=1.1*10**-3, fmt='r^',
             label="Охлаждение")
plt.grid()
plt.plot(X, Y, color="blue")
plt.plot(X, Y_back, color="red")
plt.title(r'Зависимость $ln\frac{P}{P_0}$ от $\frac{1}{T}$ ')
plt.ylabel(r'$ln\frac{P}{P_0}$')
plt.xlabel(r'$\frac{1}{T}$, $10^{-3}$ К')
plt.legend()
plt.show()



result_list_x, result_list_y = get_data('data 2.4.1', False)
result_list_x_back, result_list_y_back = get_data('data 2.4.1_back', False)

k_1, b_1 = np.polyfit(result_list_x, result_list_y, deg=1)
k_1_back, b_1_back = np.polyfit(result_list_x_back, result_list_y_back, deg=1)

X = np.arange(295.7, 310, 0.003)

Y_1 = [k_1*x + b_1 for x in X]
Y_1_back = [k_1_back*x + b_1_back for x in X]

plt.errorbar(result_list_x, result_list_y, xerr=0.1, yerr=0.14, fmt='b^', label='Нагрев')
plt.errorbar(result_list_x_back, result_list_y_back, xerr=0.1, yerr=0.14, fmt='r^', label='Охлаждение')

plt.plot(X, Y_1, color="blue")
plt.plot(X, Y_1_back, color="red")
plt.xlabel('T, K')
plt.ylabel('P, Па')
plt.legend()
plt.title('Зависимость P от T. Линейная аапроксимация')
plt.grid()

plt.show()

l_list = []
pair_list = []
sigma_list = []
sigma_k_1 = find_angle_sigma_k(result_list_x, result_list_y, k_1)
print(len(X))
I = [1, 10, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]

q = 0
for i in I:
    l_list.append(k_1*X[i]**2*8.31/Y_1[i] / 1000)
    pair_list.append( [ X[i], Y_1[i] ] )
    sigma_list.append(np.sqrt((2*0.1/X[i])**2 + (0.14*133.3/Y_1[i]) + sigma_k_1**2))
    q += 1

print(l_list)
print(pair_list)
print(sigma_list)

average = 0
for x in l_list:
    average += x
average /= len(l_list)

n = len(l_list)
sigma = 0
for i in range(n):
    sigma += sigma_list[i]**2
sigma = np.sqrt(sigma / n)


print(average, sigma)

l_list = []
pair_list = []
sigma_list = []
sigma_k_1_back = find_angle_sigma_k(result_list_x_back, result_list_y_back, k_1_back)
print(len(X))
I = [1, 10, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]

q = 0
for i in I:
    l_list.append(k_1*X[i]**2*8.31/Y_1_back[i] / 1000)
    pair_list.append( [ X[i], Y_1_back[i] ] )
    sigma_list.append(np.sqrt((2*0.1/X[i])**2 + (0.14*133.3/Y_1_back[i]) + sigma_k_1_back**2))
    q += 1

print(l_list)
print(pair_list)
print(sigma_list)

average = 0
for x in l_list:
    average += x
average /= len(l_list)

n = len(l_list)
sigma = 0
for i in range(n):
    sigma += sigma_list[i]**2
sigma = np.sqrt(sigma / n)

print(average, sigma)

k_2, b_2, c_2 = np.polyfit(result_list_x, result_list_y, deg=2)
k_2_back, b_2_back, c_2_back = np.polyfit(result_list_x_back, result_list_y_back, deg=2)

X = np.arange(295.7, 310, 0.003)

Y_1 = [k_2*x**2 + b_2*x + c_2 for x in X]
Y_1_back = [k_2_back*x**2 + b_2_back*x + c_2_back for x in X]

plt.errorbar(result_list_x, result_list_y, xerr=0.1, yerr=0.14, fmt='b^', label='Нагрев')
plt.errorbar(result_list_x_back, result_list_y_back, xerr=0.1, yerr=0.14, fmt='r^', label='Охлаждение')

plt.plot(X, Y_1, color="blue")
plt.plot(X, Y_1_back, color="red")

plt.legend()
plt.xlabel('T, K')
plt.ylabel('P, Па')
plt.grid()
plt.title('Зависимость P от T. Аппроксимация полиномом второй степени')

plt.show()

k_list = []
for i in I:
    k_list.append(2*k_2*X[i] + b_2)

n = len((k_list))
l_list = []
pair_list = []
sigma_k_2_list = []

q = 0
for i in I:
    l_list.append(8.31*X[i]**2 * k_list[q] / Y_1[i])
    pair_list.append([X[i], Y_1[i]])
    q += 1

average = 0
for x in l_list:
    average += x
average /= len(l_list)


n = len(l_list)
sigma = 0
for i in range(n):
    sigma += (l_list[i] - average)**2
sigma = np.sqrt(sigma / (n*(n-1)))


print(average / 1000, sigma / 1000)
