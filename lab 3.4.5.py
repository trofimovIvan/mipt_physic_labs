import numpy as np
import matplotlib.pyplot as plt

def plot(x_list, y_list, title):
    x_list.sort()
    y_list.sort()
    a, b, c, d, e, f, q = np.polyfit(x_list, y_list, deg=6)
   # X = np.arange(x_list[0], x_list[-1], 0.003)
   # Y = [a*x**6 + b*x**5 + c*x**4 + d*x**3 + e*x**2 + f*x for x in X]
    plt.errorbar(x_list, y_list, fmt='s', color='b')
   # plt.plot(X, Y, color='b')
    plt.xlabel('X, дел.')
    plt.ylabel('Y, дел.')
    plt.title('{}'.format(title))
    plt.grid()
    plt.show()

Fe_Si_x_list = [2.8, 2.0, 1.0, 0.6, 0.5, 0.0, 2.4, 0.4, 1.6]
Fe_Si_y_list = [2.0, 1.6, 1.0, 0.8, 0.6, 0.0, 1.8, 0.2, 1.45]

Fe_Ni_x_list = [4.2, 3.4, 3.0, 2.6, 2.0, 1.8, 1.2, 0.7, 0.4, 0.0]
F_Ni_y_list = [2.5, 2.0, 1.85, 1.6, 1.2, 1.0, 0.8, 0.3, 0.2, 0.0]

Per_x_list = [2.7, 1.8, 1.7, 1.6, 1.4, 1.2, 0.6]
Per_y_list = [2.8, 1.6, 1.4, 1.2, 0.6, 0.3, 0.05]

plot(Per_x_list, Per_y_list, 'Пермаллой')

