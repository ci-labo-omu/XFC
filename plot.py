import matplotlib.pyplot as plt
import numpy
import csv
import pylab

import numpy as np

plt.rcParams["font.size"] = 14

with open("examples/results/game10_hit_best.csv") as f:
    reader = csv.reader(f)
    y1 = [np.max(np.float_(row)) for row in reader]

with open("examples/results/game10_hit_ave.csv") as f:
    reader = csv.reader(f)
    y2 = [np.max(np.float_(row)) for row in reader]

with open("examples/results/game10_hit_worst.csv") as f:
    reader = csv.reader(f)
    y3 = [np.max(np.float_(row)) for row in reader]

with open("examples/results/game10_hit_best.csv") as f:
    reader = csv.reader(f)
    y4 = [np.min(np.float_(row)) for row in reader]

with open("examples/results/game10_hit_ave.csv") as f:
    reader = csv.reader(f)
    y5 = [np.min(np.float_(row)) for row in reader]

with open("examples/results/game10_hit_worst.csv") as f:
    reader = csv.reader(f)
    y6 = [np.min(np.float_(row)) for row in reader]

with open("examples/results/game10_hit_best.csv") as f:
    reader = csv.reader(f)
    y7 = [np.average(np.float_(row)) for row in reader]

with open("examples/results/game10_hit_ave.csv") as f:
    reader = csv.reader(f)
    y8 = [np.average(np.float_(row)) for row in reader]

with open("examples/results/game10_hit_worst.csv") as f:
    reader = csv.reader(f)
    y9 = [np.average(np.float_(row)) for row in reader]




x = np.arange(1, len(y1) + 1, 1)
plt.xlim(0, 2501)
plt.ylim(60, 180.1)
plt.rcParams["font.size"] = 16

plt.plot(x, y1, color='b')
plt.plot(x, y2,  color='r')
plt.plot(x, y3, color='g')
plt.plot(x, y4, color='b')
plt.plot(x, y5,  color='r')
plt.plot(x, y6,  color='g')
plt.plot(x, y7, label="Average of Best score select", color='orchid')
plt.plot(x, y8, label="Average of Average score select", color='black')
plt.plot(x, y9, label="Average of Worst score select", color='orange')

#pylab.subplots_adjust(right=0.7)
plt.fill_between(x, y1, y4, color="lightblue", alpha=0.5)
plt.fill_between(x, y2, y5, color="lightsalmon", alpha=0.5)
plt.fill_between(x, y3, y6, color="lightgreen", alpha=0.5)
plt.title("ランダムな環境での評価値の範囲の推移" ,y=-0.30,fontname="MS Gothic")
plt.xlabel("評価回数",fontname="MS Gothic")
plt.ylabel("評価値",fontname="MS Gothic")
plt.rcParams['figure.dpi'] = 300

#plt.title("run10")
plt.legend(loc=4)


plt.show()
