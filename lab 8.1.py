import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def T_t(x):
    return 1.09*x - 72

I = np.array([0.530, 0.569, 0.640, 0.683, 0.770, 0.879, 0.926, 0.947, 1.025, 1.130])
V = np.array([1.779, 2.127, 2.786, 3.212, 4.138, 5.421, 5.998, 6.261, 7.301, 8.800])
W = I*V
T_l = [931, 1047, 1167, 1277, 1385, 1517, 1649, 1721, 1815, 1940]
T_l = [t + 273 for t in T_l]
T = np.array([T_t(t) for t in T_l])


logT = np.log(T)
print(T)
logW = np.log(W)
print(logW)

regr = linregress(logT, logW)
x = np.arange(logT[0], logT[-1], 0.003)
y = regr.intercept + regr.slope*x
print(regr.slope)
print(regr.stderr)

plt.plot(logT, logW, 'o')
plt.xlabel(r'$\ln T$', fontsize='15')
plt.ylabel(r'$\ln W$', fontsize='15')
plt.grid()
plt.plot(x, y)
plt.show()