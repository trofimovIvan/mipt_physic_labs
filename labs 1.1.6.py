import matplotlib.pyplot as plt
import numpy as np

lg_list = [0, 0.18, 0.3, 0.42, 0.48, 0.60, 0.7, 1, 2, 3, 4, 5, 6, 6.3, 6.4, 6.7, 6.73]
k_ac_list = [0.3, 0.45, 0.5, 0.6, 0.65, 0.75, 0.95, 1, 1, 1, 1, 1, 1, 0.975, 0.95, 0.925, 0.925]
k_dc_list = [1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 0.975, 0.95, 0.925, 0.925]
lg_list_faze = [0.3, 2, 3, 4, 5.23, 5.56, 5.64, 5.73, 6, 6.18, 6.4, 6.48, 6.54]
faze_list = [0, 0, 0, 0, 0.075, 0.1, 0.125,0.25, 0.55, 0.775, 1.57, 2.08, 2.3]

def draw_ampl_frec_char():
    plt.plot(lg_list, k_ac_list, label='K ac')
    plt.plot(lg_list, k_dc_list, label='K dc')
    plt.ylabel(r'$K$', fontsize=15)
    plt.xlabel(r'$lg f$', fontsize=15)
    plt.grid()
    plt.legend(fontsize=15)

    plt.show()

def draw_fase_frec_char():
    plt.plot(lg_list_faze, faze_list)
    plt.ylabel(r'$\Delta \phi$, рад', fontsize=15)
    plt.xlabel('lg f', fontsize=15)
    plt.grid()

    plt.show()

draw_ampl_frec_char()
