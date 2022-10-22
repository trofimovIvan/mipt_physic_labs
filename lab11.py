import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

v1 = [7.142, 6.981, 6.630, 6.384, 5.919, 5.250, 4.317, 3.699, 3.088, 2.593, 1.894, 1.194, 0.516]
v1.sort()
i1 = [0.544, 0.542, 0.536, 0.532, 0.520, 0.502, 0.455, 0.378, 0.285, 0.198, 0.112, 0.057, 0.030]
i1.sort()

v2 = [7.148, 5.793, 4.944, 3.937, 3.283, 2.877, 2.393, 2.084, 1.861, 1.414, 1.174, 1.001]
i2 = [0.535, 0.510, 0.488, 0.452, 0.401, 0.369, 0.275, 0.218, 0.182, 0.119, 0.093, 0.078]
v2.sort()
i2.sort()

v3 = [7.152, 6.695, 5.715, 4.638, 3.568, 2.752, 2.024, 1.603, 1.194, 1.105]
v3.sort()
i3 = [0.526, 0.518, 0.502, 0.474, 0.428, 0.360, 0.240, 0.183, 0.127, 0.120]
i3.sort()

v4 = [7.157, 6.882, 6.069, 5.074, 4.692, 4.000, 3.434, 3.042, 2.555, 1.880, 1.317]
v4.sort()
i4 = [0.507, 0.504, 0.488, 0.466, 0.454, 0.424, 0.390, 0.359, 0.309, 0.222, 0.148]
i4.sort()

V1 = np.array(v1).reshape(-1,1)
I1 = np.array(i1).reshape(-1, 1)

V2 = np.array(v2).reshape(-1, 1)
I2 = np.array(i2).reshape(-1, 1)

V3 = np.array(v3).reshape(-1, 1)
I3 = np.array(i3).reshape(-1, 1)

V4 = np.array(v4).reshape(-1, 1)
I4 = np.array(i4).reshape(-1, 1)

I1 = I1-0.004
I2 = I2 - 0.004
I3 = I3 - 0.004
I4 = I4 - 0.004

y1 = np.sqrt(I1)
y2 = np.sqrt(I2)
y3 = np.sqrt(I3)
y4 = np.sqrt(I4)


x1lin = V1[:5]
y1lin =y1[:5]
x2lin = V2[:6]
y2lin = y2[:6]
x3lin = V3[:4]
y3lin = y3[:4]
x4lin = V4[:2]
y4lin = y4[:2]

lr1 = LinearRegression().fit(x1lin, y1lin)
lr2 = LinearRegression().fit(x2lin, y2lin)
lr3 = LinearRegression().fit(x3lin, y3lin)
lr4 = LinearRegression().fit(x4lin, y4lin)

c1 = lr1.coef_
b1 = lr1.intercept_

c2 = lr2.coef_
b2 = lr2.intercept_

c3 = lr3.coef_
b3 = lr3.intercept_

c4 = lr4.coef_
b4 = lr4.coef_

x1 = np.linspace(-1, x1lin[-1], 100)
x2 = np.linspace(-1, x2lin[-1], 100)
x3 = np.linspace(-1, x3lin[-1], 100)
x4 = np.linspace(-1, x4lin[-1], 100)


"""ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')

plt.plot(V1, y1, 'o', color='red', label='6929 A')
plt.plot(x1, c1*x1 + b1, color='red')
plt.plot(V2, y2, 'o', color='green', label='6507 A')
plt.plot(x2, c2*x2 + b2, color='green')
plt.plot(V3, y3, 'o', color='black', label='6217 A')
plt.plot(x3, c3*x3 + b3, color='black')
plt.plot(V4, y4, 'o', color='blue', label='5852 A')
plt.plot(x4, c4*x4 + b4, color='blue')
plt.xlabel('V, B', fontsize='17')
plt.ylabel(r'$\sqrt{I}$', fontsize='17')
plt.legend()
plt.grid()
plt.show()"""

V0 = np.array([b1[0]/c1[0], b2[0]/c2[0], b3[0]/c3[0], b4[0]/c4[0]]).reshape(4, 1)
lambda_list = np.array([1/6929, 1/6507, 1/6217, 1/5852])*10**10
omega = (2*np.pi*3*10**8)*lambda_list

V = V0.reshape(1, 4)
print(V)

"""lr = LinearRegression().fit(omega, V0)
x = np.linspace(omega[0], omega[-1], 100)
c = lr.coef_
b = lr.intercept_
print(c)"""
"""plt.errorbar(omega, V0, yerr=0.1,fmt='o')
plt.plot(x, c*x + b)
plt.xlabel(r'$\omega, c^{-1}$', fontsize='17')
plt.ylabel(r'$V_0$, B', fontsize='17')
plt.show()"""

res = linregress(omega, V, alternative='greater')
print(res.pvalue)
