import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

N1 = np.array([51262, 27868, 14822, 8246, 4644, 2681, 1616])
N2 = np.array([53433, 33856, 20456, 13352, 8873, 6163, 4022])
N3 = np.array([47556, 24358, 13954, 7977, 4653, 2603, 1686])
N4 = np.array([85419, 81811, 80670, 78608, 77191, 74712, 73927])
N1 = N1/10 - 12.6
N2 = N2/10 - 12.6
N3 = N3/10 - 12.6
N4 = N4/10 - 12.6

n = np.array([1, 2, 3, 4, 5, 6, 7])
x1 = (0.92*n).reshape(-1, 1)
x2 = (2.04*n).reshape(-1, 1)
x3 = (0.45*n).reshape(-1, 1)
x4 = (1.98*n).reshape(-1, 1)

lnN1 = np.log(N1)
lnN2 = np.log(N2)
lnN3 = np.log(N3)
lnN4 = np.log(N4)
lnN0 = np.log(8673.8)

y1 = (-lnN1 + lnN0).reshape(-1, 1)
y2 = (-lnN2 + lnN0).reshape(-1, 1)
y3 = (-lnN3 + lnN0).reshape(-1, 1)
y4 = (-lnN4 + lnN0).reshape(-1, 1)

lr1 = LinearRegression(fit_intercept=False).fit(x1, y1)
lr2 = LinearRegression(fit_intercept=False).fit(x2, y2)
lr3 = LinearRegression(fit_intercept=False).fit(x3, y3)
lr4 = LinearRegression(fit_intercept=False).fit(x4, y4)

a1 = lr1.coef_[0][0]
a2 = lr2.coef_[0][0]
a3 = lr3.coef_[0][0]
a4 = lr4.coef_[0][0]
print(a1, a2, a3, a4)

xx1 = np.linspace(0, 6.5, 1000)
xx2 = np.linspace(0, 15, 1000)
xx3 = np.linspace(0, 3.2, 1000)
xx4 = np.linspace(0, 15, 1000)
y_pred1 = a1*xx1
y_pred2 = a2*xx2
y_pred3 = a3*xx3
y_pred4 = a4*xx4

plt.plot(x1, y1, 'o', color='g')
plt.plot(xx1, y_pred1, color='green', label='Fe')
plt.plot(x2, y2, 'o', color='black')
plt.plot(xx2, y_pred2, color='black', label='Al')
plt.plot(x3, y3, 'o', color='red')
plt.plot(xx3, y_pred3, color='red', label='Pb')
plt.plot(x4, y4, 'o', color='blue')
plt.plot(xx4, y_pred4, color='blue', label='Пробка')
plt.legend()
plt.xlabel(r'$l, cm$', fontsize='17')
plt.ylabel(r'$\ln N_0 - \ln N$', fontsize='17')
plt.show()
