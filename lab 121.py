import matplotlib.pyplot as plt
import numpy as np

m = 0.84
d = 5.295 * 10**-1
r_max = 15.8 * 10**-1
g = 981
P_m_1 = np.sqrt(m*g*r_max**4/6)
print(P_m_1)
p_m_1 = P_m_1/(4/3 * np.pi * (d/2)**3)
print(p_m_1)
B_p = 2*P_m_1/(d/2)**3
print(B_p)
B_r = 4*np.pi*p_m_1

M = 305.235
F = M*g
F_0 = F / 1.08
print(F)
P_m_2 = np.sqrt((F*d**4)/6)
print(P_m_2)
p_m_2 = P_m_2/(4/3 * np.pi * (d/2)**3)
B_r_2 = 4*np.pi*p_m_2
print(B_r)
print(B_r_2)
B_h = np.pi**2*m*d**2/(3*0.246**2*P_m_2)
print(B_h)
print(20.251 / P_m_2)