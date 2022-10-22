import numpy as np
import matplotlib.pyplot as plt

t_0 = 8.252

def read_data():
    t_list = []
    Temp_list_0 = []
    with open('data 342', 'r') as data_file:
        data_str = data_file.readlines()
        n = len(data_str)
        for i in range(n):
            Temp_list_0.append(float(data_str[i].split()[0]) + float(data_str[i].split()[2])*24)
            t_list.append(float(data_str[i].split()[1]))

    return t_list, Temp_list_0

def f1(time_list, Temp_list):
    x_list = Temp_list
    n = len(time_list)
    y_list = [time_list[i] - t_0 for i in range(n)]

    return x_list, y_list

def f2(time_list, Temp_list):
    x_list = Temp_list
    n = len(time_list)
    y_list = [1 / (time_list[i] - t_0) for i in range(n)]

    return x_list, y_list

def plot(x_list, y_list, x_err, y_err):

    a, b = np.polyfit(x_list[6::], y_list[6::], deg=1)

    plt.errorbar(x_list, y_list, xerr= x_err, yerr= y_err, fmt='s')
    plt.grid()
    X = np.arange(x_list[3], x_list[-1], 0.003)
    Y = [a*x + b for x in X]
    plt.plot(X, Y)
    plt.show()
    print(a, b)

time_list, Temp_list = read_data()

x_list, y_list = f1(time_list, Temp_list)

x_list_2, y_list_2 = f2(time_list, Temp_list)

"""print(x_list)
print(time_list)
print(y_list)
print(y_list_2)"""

n = len(time_list)

sigma_list_1 = [2*time_list[i]*0.001 for i in range(n)]

sigma_list_2 = [2*time_list[i]*0.001 * (y_list_2[i])**2 for i in range(n)]

plot(x_list_2, y_list_2, 0.05, sigma_list_2)