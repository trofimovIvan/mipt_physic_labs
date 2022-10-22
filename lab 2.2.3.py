import numpy as np
import matplotlib.pyplot as plt

def get_u_data_temp(t):
    with open('t = {} data 223'.format(t)) as data_file:
        all_strings = data_file.readlines()
        u_e = []
        u_n = []
        for line in all_strings:
            u_e.append(float(line.split()[0]))
            u_n.append(float(line.split()[1]))
    return u_e, u_n

def get_rq_n_lists(u_e, u_n):
    n = len(u_e)
    r_list = []
    q_list = []
    for i in range(n):
        r_list.append(u_n[i] / u_e[i] * 10.00)
        q_list.append(u_n[i] * u_e[i] / 10.00)

    return r_list, q_list

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

def get_line(x_list, y_list):
    a, b =find_koef(x_list, y_list)
    X = np.arange(x_list[0], x_list[-1], 0.0003)
    Y = [a*x + b for x in X]
    return X, Y

def get_r_t_list(x_list1, y_list1, x_list2, y_list2, x_list3, y_list3, x_list4, y_list4,
               x_list5, y_list5, x_list6, y_list6):
    r_list = [find_koef(x_list1, y_list1)[1], find_koef(x_list2, y_list2)[1], find_koef(x_list3, y_list3)[1],
              find_koef(x_list4, y_list4)[1], find_koef(x_list5, y_list5)[1], find_koef(x_list6, y_list6)[1]]
    t_list = [23.0, 27.6, 35.0, 41.0, 50.0, 58.4]

    return t_list, r_list

def get_sigma_r_list(u_e, u_n, r_list):
    n = len(u_e)
    sigma_r_list = []
    for i in range(n):
        sigma_r_list.append(r_list[i] * np.sqrt((0.00001 / u_e[i])**2 + (0.00001 / u_n[i])**2))
    return sigma_r_list

def find_angle_sigma_k_b(x, y, k):
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
    x_sqr_average /= number_of_points
    y_sqr_average /= number_of_points
    x_average /= number_of_points
    y_average /= number_of_points

    sigma_k = np.sqrt(abs((y_sqr_average - y_average**2)/(x_sqr_average - x_average**2) - k**2)) \
              / np.sqrt(number_of_points)
    sigma_b = sigma_k*np.sqrt(x_sqr_average - x_average**2)
    return sigma_k, sigma_b

def get_sigma_r_list_t(x_list1, y_list1, x_list2, y_list2, x_list3, y_list3, x_list4, y_list4,
               x_list5, y_list5, x_list6, y_list6):

    sigma_r_list = [find_angle_sigma_k_b(x_list1, y_list1, find_koef(x_list1, y_list1)[0])[1],
                    find_angle_sigma_k_b(x_list2, y_list2, find_koef(x_list2, y_list2)[0])[1],
                    find_angle_sigma_k_b(x_list3, y_list3, find_koef(x_list3, y_list3)[0])[1],
                    find_angle_sigma_k_b(x_list4, y_list4, find_koef(x_list4, y_list4)[0])[1],
                    find_angle_sigma_k_b(x_list5, y_list5, find_koef(x_list5, y_list5)[0])[1],
                    find_angle_sigma_k_b(x_list6, y_list6, find_koef(x_list6, y_list6)[0])[1]]
    return sigma_r_list

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
    return k, m

def find_chi_sqr(x_list, y_list, sigma_list):
    p = len(x_list)
    chi_sqr = 0
    k, b = find_k_chi_sqr(x_list, y_list, sigma_list)
    for i in range(p):
        chi_sqr += (k*x_list[i] + b - y_list[i])**2 / sigma_list[i]**2
    return chi_sqr/(p - 2)


u_e1, u_n1 = get_u_data_temp(23.0)
u_e2, u_n2 = get_u_data_temp(27.6)
u_e3, u_n3 = get_u_data_temp(35.0)
u_e4, u_n4 = get_u_data_temp(41.0)
u_e5, u_n5 = get_u_data_temp(50.0)
u_e6, u_n6 = get_u_data_temp(58.4)

r_list1, q_list1 = get_rq_n_lists(u_e1, u_n1)
r_list2, q_list2 = get_rq_n_lists(u_e2, u_n2)
r_list3, q_list3 = get_rq_n_lists(u_e3, u_n3)
r_list4, q_list4 = get_rq_n_lists(u_e4, u_n4)
r_list5, q_list5 = get_rq_n_lists(u_e5, u_n5)
r_list6, q_list6 = get_rq_n_lists(u_e6, u_n6)

sigma_r_list1 = get_sigma_r_list(u_e1, u_n1, r_list1)
sigma_q_list1 = get_sigma_r_list(u_e1, u_n1, q_list1)

sigma_r_list2 = get_sigma_r_list(u_e2, u_n2, r_list2)
sigma_q_list2 = get_sigma_r_list(u_e2, u_n2, q_list2)
sigma_r_list3 = get_sigma_r_list(u_e3, u_n3, r_list3)
sigma_q_list3 = get_sigma_r_list(u_e3, u_n3, q_list3)
sigma_r_list4 = get_sigma_r_list(u_e4, u_n4, r_list4)
sigma_q_list4 = get_sigma_r_list(u_e4, u_n4, q_list4)
sigma_r_list5 = get_sigma_r_list(u_e5, u_n5, r_list5)
sigma_q_list5 = get_sigma_r_list(u_e5, u_n5, q_list5)
sigma_r_list6 = get_sigma_r_list(u_e6, u_n6, r_list6)
sigma_q_list6 = get_sigma_r_list(u_e6, u_n6, q_list6)


q_list1.sort()
r_list1.sort()
q_list2.sort()
r_list2.sort()
q_list3.sort()
r_list3.sort()
q_list4.sort()
r_list4.sort()
q_list5.sort()
r_list5.sort()
q_list6.sort()
r_list6.sort()

X1, Y1 = get_line(q_list1, r_list1)
a1, b1 = find_koef(q_list1, r_list1)
sigma_a1, sigma_b1 = find_angle_sigma_k_b(q_list1, r_list1, a1)

X2, Y2 = get_line(q_list2, r_list2)
a2, b2 = find_koef(q_list2, r_list2)
sigma_a2, sigma_b2  = find_angle_sigma_k_b(q_list2, r_list2, a2)

X3, Y3 = get_line(q_list3, r_list3)
a3, b3 = find_koef(q_list3, r_list3)
sigma_a3, sigma_b3  = find_angle_sigma_k_b(q_list3, r_list3, a3)

X4, Y4 = get_line(q_list4, r_list4)
a4, b4 = find_koef(q_list4, r_list4)
sigma_a4, sigma_b4  = find_angle_sigma_k_b(q_list4, r_list4, a4)

X5, Y5 = get_line(q_list5, r_list5)
a5, b5 = find_koef(q_list5, r_list5)
sigma_a5, sigma_b5 = find_angle_sigma_k_b(q_list5, r_list5, a5)

X6, Y6 = get_line(q_list6, r_list6)
a6, b6 = find_koef(q_list6, r_list6)
sigma_a6, sigma_b6 = find_angle_sigma_k_b(q_list6, r_list6, a6)

b_list = [b1, b2, b3, b4, b5, b6]
sigma_b_list = [sigma_b1, sigma_b2, sigma_b3, sigma_b4, sigma_b5, sigma_b6]

drdq_list = [a1, a2, a3, a4, a5, a6]
sigma_drdq_list = [sigma_a1, sigma_a2, sigma_a3, sigma_a4, sigma_a5, sigma_a6]

plt.errorbar(q_list1, r_list1, xerr=sigma_q_list1, fmt='s', color='blue', label=r'$t = 23.0^{o} C$')
plt.plot(X1, Y1, color='blue')
plt.errorbar(q_list2, r_list2, xerr=sigma_q_list2, fmt='s', color='red', label=r'$t = 27.6^{o} C$')
plt.plot(X2, Y2, color='red')
plt.errorbar(q_list3, r_list3, fmt='s', color='magenta', label=r'$t = 35.0^{o} C$')
plt.plot(X3, Y3, color='magenta')
plt.errorbar(q_list4, r_list4, fmt='s', color='orange', label=r'$t = 41.0^{o} C$')
plt.plot(X4, Y4, color='orange')
plt.errorbar(q_list5, r_list5, fmt='s', color='green', label=r'$t = 50.0^{o} C$')
plt.plot(X5, Y5, color='green')
plt.errorbar(q_list6, r_list6, fmt='s', color='c', label=r'$t = 58.4^{o} C$')
plt.plot(X6, Y6, color='c')
plt.grid()
plt.xlabel('Q, Дж', fontsize='17')
plt.ylabel('R, Ом', fontsize='17')
plt.title('R(Q)', fontsize='20')
plt.legend(fontsize='12')
plt.show()

t_list, r_t_list = get_r_t_list(q_list1, r_list1, q_list2, r_list2, q_list3, r_list3, q_list4, r_list4,
                                q_list5, r_list5, q_list6, r_list6)

sigma_r_list = get_sigma_r_list_t(q_list1, r_list1, q_list2, r_list2, q_list3, r_list3, q_list4, r_list4,
                                q_list5, r_list5, q_list6, r_list6)

X_T, Y_R = get_line(t_list, r_t_list)
k, b =find_koef(t_list, r_t_list)
sigma_k, sigma_b = find_angle_sigma_k_b(t_list, r_t_list, k)

alpha = k / b
sigma_alpha = alpha * np.sqrt((sigma_k / k)**2 + (sigma_b / b)**2)



plt.errorbar(t_list, r_t_list, xerr=0.05, yerr=0.001, fmt='s', color='blue')
plt.plot(X_T, Y_R, color='blue')
plt.xlabel(r'$t, ^o C$', fontsize='17')
plt.ylabel('R, Ом', fontsize='17')
plt.title('График R(t)', fontsize='20')
plt.grid()
plt.show()


kappa_list = [k / drdq_list[i] * 5.30 / (2*np.pi * 0.365) for i in range(len(drdq_list))]
sigma_kappa_list = [np.sqrt((sigma_k / k)**2 + (sigma_drdq_list[i] / drdq_list[i])**2 + (2/365)**2) * kappa_list[i]
                    for i in range(len(kappa_list))]

t_list_tabl = [17, 27, 37, 67]
kappa_list_tabl = [0.02485, 0.02553, 0.02621, 0.02836]



plt.errorbar(t_list, kappa_list, xerr=0.05, yerr=sigma_kappa_list, fmt='s',  color='blue',
             label='Экспериментальные точки')
plt.errorbar(t_list_tabl, kappa_list_tabl, fmt='s', color='red', label='Теоретические точки', linestyle = '--')
plt.legend(fontsize='12')
plt.xlabel(r'$t ^o C$', fontsize='17')
plt.ylabel(r'$\kappa, \frac{Вт}{м * ^o C}$', fontsize='17')
plt.title(r'Зависимость $\kappa (t)$', fontsize='20')
plt.grid()
plt.show()


kappa_log = [np.log(kappa_list[i]) for i in range(0, len(kappa_list))]
t_list_log = [np.log(t_list[i]) for i in range(0, len(t_list))]
t_list_tabl_log = [np.log(t_list_tabl[i]) for i in range(len(t_list_tabl))]
kappa_list_tabl_log = [np.log(kappa_list_tabl[i]) for i in range(len(kappa_list_tabl))]

sigma_kappa_log_list = [sigma_kappa_list[i] / kappa_list[i] for i in range(0, len(kappa_list))]



k_ln, b_ln = find_k_chi_sqr(t_list_log, kappa_log, sigma_kappa_log_list)
print(k_ln)

X_LN_T = np.arange(t_list_log[0], t_list_log[-1], 0.003)
Y_LN_K = [k_ln*x + b_ln for x in X_LN_T]

k_ln_tabl, b_ln_tabl = find_koef(t_list_tabl_log, kappa_list_tabl_log)
print(k_ln_tabl)
X_LN_T_tabl = np.arange(t_list_tabl_log[0], t_list_tabl_log[-1], 0.003)
Y_LN_K_tabl = [k_ln_tabl*x + b_ln_tabl for x in X_LN_T_tabl]

plt.errorbar(t_list_log, kappa_log, yerr=sigma_kappa_log_list, fmt = 's', color='blue', label='Экспериментальные точки')
plt.errorbar(t_list_tabl_log, kappa_list_tabl_log, fmt='s', color='red', label='Теоретические точки')
plt.plot(X_LN_T, Y_LN_K, color='blue')
plt.plot(X_LN_T_tabl, Y_LN_K_tabl, color='red')
plt.xlabel('ln t', fontsize='17')
plt.ylabel(r'$ln \kappa$', fontsize='17')
plt.title(r'Зависимость $ln \kappa (ln t)$', fontsize='20')
plt.legend(fontsize='12')
plt.grid()
plt.show()

print(find_chi_sqr(t_list_log, kappa_log, sigma_kappa_log_list))
