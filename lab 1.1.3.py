import math
import matplotlib.pyplot as plt
import numpy as np

#определяю нормальное распределение
def gauss_func(x, sr, sigm):
    f_y = []
    for i in range(len(x)):
        y = 2*(math.exp(-(x[i] - sr) ** 2 / (2 * sigm**2))) / (math.sqrt(2 * math.pi) * sigm)
        f_y.append(y)
    return f_y

data_resistors_file = open('data_resistors', 'r')
data_resistors_list = []
y_list = []
m = 20

#считываю данные, перевожу их в float
for now in data_resistors_file:
    if float(now) <= 5.21:
        data_resistors_list.append(float(now))
data_resistors_list.sort()

#вычисляю дельта R
delta_R = (data_resistors_list[-1] - data_resistors_list[0]) / m

#вычисляю среднее сопротивление
sum = 0
for now in data_resistors_list:
    sum += now
r_sr = sum / len(data_resistors_list)
#вычисляю сколько резисторов попадают в интервал
r_i = data_resistors_list[0]
x_list=[]
x_coord= delta_R/2 + data_resistors_list[0]
for i in range(m):
    n = 0
    x_list.append(x_coord)
    for i in range(len(data_resistors_list)):
        if abs(r_i - data_resistors_list[i]) <=  delta_R:
            n += 1
    y =  n / (len(data_resistors_list) * delta_R)
    print(y)
    x_coord+=delta_R
    r_i += delta_R
    y_list.append(y)
print(y_list)
print(delta_R)
#вычисляю погрешность
r_i_minus_r_sr_sq = 0
for i in range(len(data_resistors_list)):
    r_i_minus_r_sr_sq += (data_resistors_list[i] - r_sr)**2
sigma = math.sqrt(r_i_minus_r_sr_sq/len(data_resistors_list))

#рисую график и гистограмму
ax = plt.subplot()
ax.plot(data_resistors_list, gauss_func(data_resistors_list, r_sr, sigma), color='red')
ax.set_xlabel('R, Ом', fontsize=15)
ax.set_ylabel(r'$\omega$', fontsize=15)
width = 0.01
plt.bar(x_list, y_list, delta_R)
plt.text(r_sr, 0.005, r'$R_s$', fontsize=15)
plt.text(r_sr + sigma, 0.01, r'$R_s + \sigma$', fontsize=12)
plt.text(r_sr - sigma, 0.01, r'$R_s - \sigma$', fontsize=12)
plt.text(r_sr + 2*sigma, 0.01, r'$R_s + 2\sigma$', fontsize=12)
plt.text(r_sr - 2*sigma, 0.01, r'$R_s - 2\sigma$', fontsize=12)
sigma = round(sigma*10**3) / 10**3
r_sr = round(r_sr*10**3) / 10**3
plt.text(5.14, 26, r'$R_s = {r} Ом$'.format(r=r_sr), fontsize=12)
plt.text(5.14, 24, r'$\sigma = {s} Ом$'.format(s=sigma), fontsize=12)

#вычисляю долю попавших в сигма-окрестность
n_sigma = 0
for i in range(len(data_resistors_list)):
    if r_sr - sigma <= data_resistors_list[i] <= r_sr + sigma:
        n_sigma += 1
n_sigma_part = n_sigma / len(data_resistors_list)

n_2sigma = 0
for i in range(len(data_resistors_list)):
    if r_sr - 2*sigma <= data_resistors_list[i] <= r_sr + 2*sigma:
        n_2sigma += 1
n_2sigma_part = n_2sigma / len(data_resistors_list)

print(n_sigma_part, n_2sigma_part)
data_resistors_file.close()
plt.show()
