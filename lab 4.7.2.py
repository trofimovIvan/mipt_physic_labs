import numpy as np
import matplotlib.pyplot as plt


r_sqr_list = [5, 7.5, 9, 10.5, 11.4]
m_list = [1, 2, 3, 4, 5]
for i in range(len(r_sqr_list)):
    r_sqr_list[i] = (r_sqr_list[i]/2)**2

[a, b], pcov = np.polyfit(m_list, r_sqr_list, deg=1, cov=True)
a_err, b_err = np.sqrt(np.diag(pcov))
print(a, a_err)
print(b, b_err)

X = np.arange(m_list[0], m_list[-1], 0.01)
Y = [a*x + b for x in X]

"""plt.errorbar(m_list, r_sqr_list,yerr=0.2, fmt='s', color='red')
plt.plot(X, Y)
plt.xlabel('m', fontsize='15')
plt.ylabel(r'$r^2 см^2$', fontsize='15')
plt.title(r'График зависимости $r^2$ от m', fontsize='20')
plt.grid()
plt.show()"""

dvn = 0.63*10**-6 * (2.29 * 0.8)**2 / (a * 0.026 * 10**-4)
print(dvn)

