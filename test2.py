import numpy as np
import matplotlib.pyplot as plt


pop = np.random.randint(0,100,100)
n, bin, patches = plt.hist(pop, bins=20)
plt.show()