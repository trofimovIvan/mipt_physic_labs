import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
"""P = [40, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 360, 385, 440,
     460, 480, 510, 535, 560, 585, 610, 635, 660, 680, 710, 760]
I = [49, 28, 65, 106, 150, 190, 234, 274, 321, 364, 416, 464, 512, 565, 632, 679, 817,
     850, 900, 960, 1000, 1030, 1040, 1033, 1025, 1020, 1018, 1007, 995]

X1 = np.array(P[:P.index(535)]).reshape(-1, 1)
Y1 = np.array(I[:P.index(535)]).reshape(-1, 1)
X2 = np.array(P[P.index(535):]).reshape(-1, 1)
Y2 = np.array(I[P.index(535):]).reshape(-1, 1)
lr1 = LinearRegression()
lr1.fit(X1, Y1)
print(lr1.coef_, lr1.intercept_)
lr2 = LinearRegression()
lr2.fit(X2, Y2)
print(lr2.coef_, lr2.intercept_)
P = np.array(P).reshape(-1, 1)
plt.plot(P, I, 'ro')
plt.plot(P, lr1.predict(P), color='blue')
plt.plot(P, lr2.predict(P), color='blue')
plt.xlabel('P, Торр', fontsize='17')
plt.ylabel('I, пА', fontsize='17')
plt.grid()
plt.show()"""

def f(x, a, b, x0):
    return a*np.exp(-b*(x-x0)**2)

N = np.array([3254, 3254, 2938, 2694, 2492, 2277, 1889, 1421, 1258, 827, 473, 245, 162, 93,
              101, 85, 53, 15, 2, 3, 3.5, 4, 7])
N = N/10
P = np.array([50, 60, 70, 90, 100, 110, 130, 150, 160, 180, 200, 220, 240, 250 ,260, 280, 290, 300, 320,
              340, 350, 400, 450])

popt, pcov = curve_fit(f, P, N, [300, 0.00250, 50])
print(popt)


"""pp = np.polyfit(P, N, deg=5)
X = np.arange(P[0], P[-1], 0.3)
p = np.poly1d(pp)
p2 = np.poly1d([p.c[0]*5, p.c[1]*4, p.c[2]*3, p.c[3]*2, p.c[4]*1, 0])"""
X = np.arange(P[0], P[-1], 0.3)
X2 = np.arange(P[0], 230, 0.3)
plt.plot(P, N, 'ro')
"""plt.plot(X, p(X), color='red')
plt.plot(X, abs(p2(X)), color='blue')"""
plt.plot(X, [f(x, popt[0], popt[1], popt[2]) for x in X], color='red', label='N = 325*exp(-0.00007(x-40)^2) ')
plt.plot(X, [popt[0]*popt[1]*(x-popt[2])*2*150*np.exp(-popt[1]*(x-popt[2])**2) for x in X], color='blue',
         label=r'$\frac{dN}{dP}$')
plt.vlines(123, 0, 350, linestyles='-.', color='blue')
plt.plot(X2, [f(123, popt[0], popt[1], popt[2]) - popt[0]*popt[1]*(123-popt[2])*2*np.exp(-popt[1]*(123-popt[2])**2)*
             (x-123) for x in X2])
plt.legend()
plt.xlabel('P, Торр', fontsize="17")
plt.ylabel(r'$N, c^{-1}$', fontsize='17')
plt.grid()
plt.show()