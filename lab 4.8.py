import numpy as np
import matplotlib.pyplot as plt

def get_values_list(list, kappa):
    for i in range(len(list)):
        list[i] = list[i]*kappa
    return list

x_list = [5, 7, 9, 10, 11, 13, 15, 17, 19, 21]
sigma_x = 0.2
I_list = [33, 36, 38, 38, 39, 41, 41, 42, 42, 42]
sigma_I = 1
U_R_list = [73, 79, 84, 84, 86, 89, 90, 92, 92, 93]
sigma_U_R = 1
U_L_list = [80, 72, 65, 62, 60, 56, 52, 50, 47, 45]
sigma_U_L = 1
U_R_L_list = [117, 115, 113, 111, 112, 112, 111, 110, 110, 109]
sigma_U_R_L = 1
P_L_list = [47, 43, 40, 39, 38, 36, 34, 33, 32, 30]
sigma_P_L = 0.5
kappa_R = 1
kappa_L = 1
kappa_I = 0.25
kappa_R_L = 1
kappa_P_L = 0.25

sigma_I = kappa_I*sigma_I
sigma_U_R *= kappa_R
sigma_U_L *= kappa_L
sigma_P_L *= kappa_P_L
sigma_U_R_L *= kappa_R_L

I_list = get_values_list(I_list, kappa_I)
U_R_list = get_values_list(U_R_list, kappa_R)
U_R_L_list = get_values_list(U_R_L_list, kappa_R_L)
P_L_list = get_values_list(P_L_list, kappa_P_L)

r_L_list = [P_L_list[i] / I_list[i]**2 for i in range(len(I_list))]
sigma_r_l_list = [r_L_list[i]*np.sqrt((sigma_P_L/P_L_list[i])**2 + (2*sigma_I/I_list[i])**2)
                  for i in range(len(r_L_list))]
L_list = [np.sqrt((U_L_list[i]/I_list[i])**2 - r_L_list[i]**2)/50/(2*np.pi) for i in range(len(r_L_list))]
sigma_U_L_div_I_list = [U_L_list[i]/I_list[i]*np.sqrt((sigma_U_L/U_L_list[i])**2 + (sigma_I/I_list[i])**2)
                        for i in range(len(r_L_list))]
sigma_L_list = [np.sqrt((U_L_list[i]*sigma_U_L_div_I_list[i])**2/((U_L_list[i]/I_list[i])**2 - r_L_list[i]**2)
                + (r_L_list[i]*sigma_r_l_list[i])**2 / ((U_L_list[i]/I_list[i])**2 - r_L_list[i]**2))/(50*2*np.pi)
                for i in range(len(r_L_list))]


with open('data 4.8.txt', 'w') as res_file:
    for i in range(len(x_list)):
        print(x_list[i], ' ', I_list[i], ' ', U_R_list[i], ' ', U_L_list[i], ' ', U_R_L_list[i],
              ' ', P_L_list[i], ' ', round(r_L_list[i], 2), ' ', round(sigma_r_l_list[i], 2), ' ',
              round(1000*L_list[i], 1), ' ', round(1000*sigma_L_list[i], 1), file=res_file)

plt.errorbar(x_list, r_L_list, fmt='s', yerr=sigma_r_l_list, label=r'Сопротивление r')
plt.errorbar(x_list, L_list, fmt='s', yerr=sigma_L_list, label=r'Индуктивность L')
plt.grid()
plt.legend(fontsize='15')
plt.xlabel('x, мм')
plt.ylabel( r'$r_L, L, Ом, Гн.$')
plt.title('Зависимость сопротивления и '
          'индуктивности катушки от положения сердчечника', fontsize='17')
plt.show()
