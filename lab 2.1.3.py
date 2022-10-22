import numpy as np
import matplotlib.pyplot as plt

list_of_c_and_sigma = []
list_of_sigma_c = []

def find_angle_sigma_k(x, y, k):
    number_of_points = len(x)
    x_sqr_average = 0
    y_sqr_average = 0
    for i in range(number_of_points):
        x_sqr_average += x[i]**2
        y_sqr_average += y[i]**2
    x_sqr_average /= number_of_points
    y_sqr_average /= number_of_points

    sigma_k = (np.sqrt(abs((y_sqr_average / x_sqr_average) - k**2)))/ np.sqrt(number_of_points)
    return sigma_k

def find_koef(x_list, y_list):
    n = len(x_list)
    znam = 0
    chisl = 0
    for i in range(n):
        chisl += x_list[i]*y_list[i]
        znam += x_list[i]**2
    return chisl / znam

def show_graph_and_koef(k_list, data_list, data_list_back, frequency):
    k = find_koef(k_list, data_list)
    k_back = find_koef(k_list, data_list_back)
    sigma_k = find_angle_sigma_k(k_list, data_list, k)
    sigma_k_back = find_angle_sigma_k(k_list, data_list_back, k_back)
    c = 2*k*frequency
    sigma_c = c*sigma_k / k
    c_back = 2*k_back*frequency
    sigma_c_back = c_back*sigma_k_back / k_back
    print(k, k_back)
    list_of_sigma_c.append([sigma_c, sigma_c_back])
    print('straight = ', c, sigma_c, 'back = ', c_back, sigma_c_back)
    list_of_c_and_sigma.append([c, c_back])
    plt.errorbar(k_list, data_list, yerr=0.7, label='Движение трубки вперед', marker="^")
    plt.errorbar(k_list, data_list_back, yerr=0.7, label = 'Движение трубки назад', marker=".")
    plt.xlabel('k')
    plt.ylabel(r'$\Delta L$, мм')
    plt.grid()
    plt.title('f = {} кГц'.format(frequency))
    plt.legend()
    plt.show()

def sigma_sr(data_list, average):
    n = len(data_list)
    sigma = 0
    for i in range(n):
        sigma += (data_list[i][0] - average)**2 + (data_list[i][1] - average)**2
    sigma = np.sqrt(sigma / (2*n*(2*n - 1)))
    return sigma

frequency_1 = 2.94
k_air_1 = [1, 2, 3]
data_air_f_1_straight = [59, 117, 176]
data_air_f_1_back = [59, 118, 177]

frequency_2 = 3.0
k_air_2 = [1, 2, 3]
data_air_f_2_straight = [57, 115, 173]
data_air_f_2_back = [58, 115, 172]

frequency_3 = 3.2
k_air_3 = [1, 2, 3]
data_air_f_3_straight = [54, 108, 162]
data_air_f_3_back = [55, 109, 163]

frequency_4 = 3.44
k_air_4 = [1, 2, 3]
data_air_f_4_straight = [50, 101, 150]
data_air_f_4_back = [50, 100, 149]

frequency_5 = 3.59
k_air_5 = [1, 2, 3, 4]
data_air_f_5_straight = [49, 97, 145, 193]
data_air_f_5_back = [49, 97, 144, 193]

"""show_graph_and_koef(k_air_1, data_air_f_1_straight, data_air_f_1_back, frequency_1)
show_graph_and_koef(k_air_2, data_air_f_2_straight, data_air_f_2_back, frequency_2)
show_graph_and_koef(k_air_3, data_air_f_3_straight, data_air_f_3_back, frequency_3)
show_graph_and_koef(k_air_4, data_air_f_4_straight, data_air_f_4_back, frequency_4)
show_graph_and_koef(k_air_5, data_air_f_5_straight, data_air_f_5_back, frequency_5)

average = 0

for i in range(len(list_of_c_and_sigma)):
    average += list_of_c_and_sigma[i][0] + list_of_c_and_sigma[i][1]

average /= (2*len(list_of_c_and_sigma))

sigma_c = 0

for i in range(len(list_of_sigma_c)):
    sigma_c += list_of_sigma_c[i][0]**2 + list_of_sigma_c[i][1]**2

sigma_c = np.sqrt(sigma_c) / (2*len(list_of_sigma_c))
print(average, sigma_c)"""



frequency_1 = 2.92
k_co2_1 = [1, 2, 3, 4]
data_co2_f_1_straight = [45, 91, 138, 185]
data_co2_f_1_back = [45, 92, 138, 185]

frequency_2 = 3.2
k_co2_2 = [1, 2, 3, 4]
data_co2_f_2_straight = [41, 84, 125, 167]
data_co2_f_2_back = [42, 84, 126, 167]

frequency_3 = 3.42
k_co2_3 = [1, 2, 3, 4]
data_co2_f_3_straight = [39, 79, 119, 159]
data_co2_f_3_back = [38, 78, 118, 157]

frequency_4 = 3.65
k_co2_4 = [1, 2, 3, 4]
data_co2_f_4_straight = [37, 74, 112, 149]
data_co2_f_4_back = [38, 74, 112, 147]

frequency_5 = 3.82
k_co2_5 = [1, 2, 3, 4]
data_co2_f_5_straight = [34, 71, 108, 143]
data_co2_f_5_back = [34, 71, 108, 144]

frequency_6 = 2.74
k_co2_6 = [1, 2, 3, 4]
data_co2_f_6_straight = [49, 99, 148, 198]
data_co2_f_6_back = [49, 98, 147, 198]

show_graph_and_koef(k_co2_1, data_co2_f_1_straight, data_co2_f_1_back, frequency_1)
show_graph_and_koef(k_co2_2, data_co2_f_2_straight, data_co2_f_2_back, frequency_2)
show_graph_and_koef(k_co2_3, data_co2_f_3_straight, data_co2_f_3_back, frequency_3)
show_graph_and_koef(k_co2_4, data_co2_f_4_straight, data_co2_f_4_back, frequency_4)
show_graph_and_koef(k_co2_5, data_co2_f_5_straight, data_co2_f_5_back, frequency_5)
show_graph_and_koef(k_co2_6, data_co2_f_6_straight, data_co2_f_6_back, frequency_6)

average = 0

for i in range(len(list_of_c_and_sigma)):
    average += list_of_c_and_sigma[i][0] + list_of_c_and_sigma[i][1]

average /= (2*len(list_of_c_and_sigma))

sigma_c = 0

for i in range(len(list_of_sigma_c)):
    sigma_c += list_of_sigma_c[i][0]**2 + list_of_sigma_c[i][1]**2

sigma_c = np.sqrt(sigma_c) / (2*len(list_of_sigma_c))
print(average, sigma_c)
