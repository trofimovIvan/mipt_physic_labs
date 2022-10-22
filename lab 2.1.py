import numpy as np
import matplotlib.pyplot as plt

I1 = np.array([0.0202, 0.071, 0.078, 0.091, 0.100, 0.103, 0.104, 0.103, 0.102, 0.101, 0.098, 0.093,
               0.089, 0.081, 0.074, 0.067, 0.061, 0.064, 0.073, 0.090, 0.104, 0.121, 0.154, 0.157, 0.168,
               0.171, 0.174, 0.175, 0.175, 0.175, 0.174, 0.170, 0.167, 0.163, 0.158, 0.152, 0.151, 0.153,
               0.156, 0.160, 0.168, 0.189, 0.212])
V1 = np.array([3.77, 11.92, 13.37, 15.55, 17.32, 18.15, 18.73, 19.66, 19.78, 20.08, 21.61, 22.17,
               22.25, 22.37, 22.57, 22.92, 23.78, 24.56, 25.50, 27.19, 28.71, 30.46, 34.00, 34.25, 35.52,
               35.90, 36.42, 36.92, 37.42, 38.16, 38.52, 39.66, 40.02, 40.48, 41.16, 42.71, 45.50, 46.22,
               47.31, 48.31, 50.17, 53.79, 57.74])

I12 = np.array([0.188, 0.172, 0.162, 0.158, 0.153, 0.152, 0.151, 0.155, 0.157, 0.169, 0.175, 0.177, 0.179,
                0.175, 0.163, 0.140, 0.111, 0.091, 0.085, 0.075, 0.069, 0.063, 0.068, 0.102, 0.105, 0.107,
                0.108, 0.104, 0.102, 0.091, 0.080, 0.052])

V12 = np.array([53.57, 50.86, 48.73, 47.43, 46.22, 45.37, 44.32, 42.30, 41.64, 40.06, 39.27, 37.90, 36.76,
                35.56, 34.05, 31.70, 28.86, 26.91, 26.32, 25.42, 24.89, 23.92, 22.93, 21.60, 20.15, 19.52,
                18.44, 17.40, 16.97, 14.94, 13.17, 8.86])

I2 = np.array([0.04, 0.084, 0.089, 0.092, 0.096, 0.098, 0.095, 0.093, 0.088, 0.078, 0.051, 0.036, 0.035,
               0.040, 0.047, 0.058, 0.067, 0.080, 0.098, 0.127, 0.140, 0.143, 0.147, 0.147, 0.147, 0.147, 0.141,
               0.136, 0.123, 0.111, 0.116, 0.111, 0.113, 0.117, 0.123, 0.134, 0.141, 0.165])

V2 = np.array([8.84, 15.84, 16.69, 17.27, 18.03, 20.30, 21.12, 21.38, 22.10, 22.67, 23.10, 24.01, 25.70, 26.42,
               27.07, 27.94, 28.69, 29.80, 31.42, 34.12, 35.42, 35.89, 36.58, 37.60, 37.92, 38.60, 40.03, 40.66,
               42.55, 46.18, 44.19, 47.00, 48.53, 49.60, 51.20, 53.41, 54.72, 59.17])


I3 = np.array([0.021, 0.053, 0.072, 0.085, 0.090, 0.093, 0.092, 0.088, 0.083, 0.037, 0.022, 0.010, 0.011, 0.012,
               0.014, 0.020, 0.022, 0.040, 0.075, 0.094, 0.100, 0.115, 0.116, 0.117, 0.116, 0.112, 0.103, 0.099,
               0.088, 0.077, 0.076, 0.076, 0.079, 0.083, 0.091, 0.104, 0.111])

V3 = np.array([8.05, 12.41, 15.40, 17.51, 18.55, 20.65, 21.18, 21.90, 22.52, 23.60, 24.23, 25.83, 26.10, 26.66,
               27.00, 27.96, 28.17, 29.64, 32.54, 34.28, 34.78, 36.83, 37.72, 38.67, 39.34, 40.05, 41.41, 42.26,
               44.26, 46.80, 50.66, 50.96, 51.63, 52.92, 54.71, 57.05, 58.83])

plt.plot(V1, I1, 'o', color='red', label='V = 4 B, прямо')
plt.plot(V12, I12, 'o', color='coral', label='V = 4 B, обратно')
plt.plot(V2, I2, 'o', color='blue', label='V = 6 B')
plt.plot(V3, I3, 'o', color='green', label='V = 8 B')
plt.xlabel('V, B',fontsize='17')
plt.ylabel('I, mA', fontsize='17')
plt.grid()
plt.legend()
plt.show()