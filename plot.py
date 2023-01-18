import matplotlib.pyplot as plt
import numpy
import csv

import numpy as np
plt.rcParams["font.family"] = "Times new roman"
plt.rcParams["font.size"] = 14
with open("results/new1.csv") as f:
    reader = csv.reader(f)
    y = [np.max(np.float_(row)) for row in reader]
    x = np.arange(1, len(y)+1, 1)

    plt.plot(x, y)
    plt.show()