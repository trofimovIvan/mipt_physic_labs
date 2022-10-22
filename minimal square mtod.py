import numpy as np

def find_angle_koef_and_sigma_k(x, y):
    number_of_points = len(x)
    x_sqr_average = 0
    y_sqr_average = 0
    x_y_multiply = 0
    for i in range(number_of_points):
        x_sqr_average += x[i]**2
        y_sqr_average += y[i]**2
        x_y_multiply += x[i]*y[i]
    x_sqr_average /= number_of_points
    y_sqr_average /= number_of_points
    x_y_multiply /= number_of_points
    k = x_y_multiply / x_sqr_average

    sigma_k = (np.sqrt(y_sqr_average / x_sqr_average - k**2)) / np.sqrt(number_of_points)
    return sigma_k


