import matplotlib.pyplot as plt
import numpy
import csv

import numpy as np
plt.rcParams["font.family"] = "Times new roman"
plt.rcParams["font.size"] = 14
with open("run10_worst2.csv") as f:
    reader = csv.reader(f)
    y = [np.average(np.float_(row)) for row in reader]
    print(y)
    x = np.arange(1, 251, 1)
    plt.title("10 Trial Worst select")
    plt.plot(x, y)
    plt.show()