import numpy as np
import matplotlib.pyplot as plt


my_data_file = open('C:/Users/Home/Desktop/2021.03.23 Трофимов Б02-015.txt', 'r')

def read_data_strings(a, b, file_to_read):
    data = []
    for line in file_to_read:
        data.append(line.split()[6])
    n = len(data)
    print(n)
    for i in range(n):
        if data[i].find(',') != -1:
            data[i] = data[i].replace(',', '.')

    res_data = []
    time_list = []
    time = 0
    n = len(data)
    for i in range(a, b):

        res_data.append(np.log(float(data[i])))
        time_list.append(time)
        time += 2

    return res_data, time_list

def read_data_strings_1(a, b, file_to_read):
    data = []
    for line in file_to_read:
        data.append(line.split()[6])
    n = len(data)
    print(n)
    for i in range(n):
        if data[i].find(',') != -1:
            data[i] = data[i].replace(',', '.')

    res_data = []
    time_list = []
    time = 0
    n = len(data)
    for i in range(a, b):

        res_data.append((float(data[i])))
        time_list.append(time)
        time += 2

    return res_data, time_list

def read_data_strings_2(a, b, file_to_read):
    data = []
    data_2 = []
    for line in file_to_read:
        data.append(line.split()[4])
        data_2.append(line.split()[6])
    n = len(data)
    print(n)
    for i in range(n):
        if data_2[i].find(',') != -1:
            data[i] = data[i].replace(',', '.')
            data_2[i] = data_2[i].replace(',', '.')

    res_data_1 = []
    res_data_2 = []
    n = len(data)
    for i in range(a, b):
        res_data_1.append((float(data[i])))
        res_data_2.append(float(data_2[i]))

    res_data_1.sort()
    res_data_2.sort()

    res_data_1_f = []
    res_data_2_f = []

    for i in range(1, len(res_data_2)):
        if res_data_1[i] > res_data_1[i - 1]:
            res_data_1_f.append(res_data_1[i])
            res_data_2_f.append(res_data_2[i])

    return res_data_1_f, res_data_2_f

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

data, time = read_data_strings(1536, 1558, my_data_file)
data_const = []
time_const = []
time_q = 42


a, b = find_koef(time, data)

X = np.arange(time[0], time[-1], 0.003)
Y = [a*x + b for x in X]

plt.errorbar(time, data, fmt='s', color='blue')
plt.plot(X, Y, color='blue')
plt.grid()
plt.xlabel('time, с', fontsize='17')
plt.ylabel(r'$ln \frac{P}{P_0}$', fontsize='17')
plt.show()

sigma_a = find_angle_sigma_k(time, data, a)

print(-1805*a * 10**-6 * 3600, 1805*sigma_a*10**(-6)*3600)

my_data_file = open('C:/Users/Home/Desktop/2021.03.23 Трофимов Б02-015.txt', 'r')

data, time = read_data_strings(1689, 1803, my_data_file)

data_1 = [data[i] for i in range(50, len(data))]
time_1 = [time[i] for i in range(50, len(time))]

a, b = find_koef(time_1, data_1)
X = np.arange(0, 240, 0.003)
Y = [a*x + b for x in X]

plt.errorbar(time, data, fmt='s',color='blue')
plt.plot(X, Y, color='blue')
plt.xlabel('time, с', fontsize='17')
plt.ylabel(r'$ln \frac{P}{P_0}$', fontsize='17')
plt.grid()
plt.show()

sigma_a = find_angle_sigma_k(time_1, data_1, a)

print(a)
print(-a*1230, sigma_a*1230)

my_data_file = open('C:/Users/Home/Desktop/2021.03.23 Трофимов Б02-015.txt', 'r')

data, time = read_data_strings(1947, 1985, my_data_file)

data_1 = [data[i] for i in range(0, len(data))]
time_1 = [time[i] for i in range(0, len(time))]

a, b = find_koef(time_1, data_1)
X = np.arange(0, 75, 0.003)
Y = [a*x + b for x in X]

plt.errorbar(time, data, fmt='s',color='blue')
plt.plot(X, Y, color='blue')
plt.xlabel('time, с', fontsize='17')
plt.ylabel(r'$ln \frac{P}{P_0}$', fontsize='17')
plt.grid()
plt.show()

sigma_a = find_angle_sigma_k(time_1, data_1, a)

print(-a*1230, sigma_a*1230)


my_data_file = open('C:/Users/Home/Desktop/2021.03.23 Трофимов Б02-015.txt', 'r')

data, time = read_data_strings_1(2008, 2045, my_data_file)
a, b = find_koef(time, data)
X = np.arange(0, 72, 0.003)
Y = [a*x + b for x in X]


plt.errorbar(time, data, fmt='s', color='blue')
plt.plot(X, Y, color='blue')
plt.grid()
plt.xlabel('time, c', fontsize='17')
plt.ylabel('P, mbar', fontsize='17')
plt.show()

sigma_a = find_angle_sigma_k(time, data, a)

sigma_res = np.sqrt( np.sqrt((35/1200)**2 + (sigma_a / a)**2))*1230*a

print(-1230*a, sigma_res)

my_data_file = open('C:/Users/Home/Desktop/2021.03.23 Трофимов Б02-015.txt', 'r')

data_w, data_p = read_data_strings_2(2680, 2822, my_data_file)

data_w_1 = [data_m for data_m in data_w if data_m < 21]
data_w_2 = [data_m for data_m in data_w if data_m >= 21]
data_p_1 = [data_p[i] for i in range(len(data_p)) if data_w[i] < 21]
data_p_2 = [data_p[i] for i in range(len(data_p)) if data_w[i] >= 21]

a1, b1 = find_koef(data_p_1, data_w_1)
a2, b2 = find_koef(data_p_2, data_w_2)

X_1 = np.arange(data_p[0], data_p_1[-1], 0.00005)
Y_1 = [a1*x + b1 for x in X_1]

X_2 = np.arange(data_p_2[0], data_p_2[-1], 0.00005)
Y_2 = [a2*x + b2 for x in X_2]

plt.errorbar(data_p, data_w, fmt='s', color='blue')
#plt.plot(X_1, Y_1)
plt.plot(X_2, Y_2, color='blue')
plt.xlabel('P, мбар', fontsize='17')
plt.ylabel('W, Вт', fontsize='17')

plt.grid()
plt.show()
