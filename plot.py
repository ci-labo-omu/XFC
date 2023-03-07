import matplotlib.pyplot as plt
import numpy
import csv
import pylab

import numpy as np

plt.rcParams["font.size"] = 14

with open("Officials.csv") as f:
    reader = csv.reader(f)
    y1 = [np.max(np.float_(row)) for row in reader]
    x1 = np.arange(1, len(y1)+1, 1)
with open("Official_both.csv") as f:
    reader = csv.reader(f)
    y2 = [np.max(np.float_(row)) for row in reader]
    x2 = np.arange(1, len(y2)+1, 1)
with open("Official_out.csv") as f:
    reader = csv.reader(f)
    y3 = [np.max(np.float_(row)) for row in reader]
    x3 = np.arange(1, len(y3)+1, 1)
with open("Officials.csv") as f:
    reader = csv.reader(f)
    y4 = [np.min(np.float_(row)) for row in reader]

with open("Official_both.csv") as f:
    reader = csv.reader(f)
    y5 = [np.min(np.float_(row)) for row in reader]

with open("Official_out.csv") as f:
    reader = csv.reader(f)
    y6 = [np.min(np.float_(row)) for row in reader]

with open("Officials.csv") as f:
    reader = csv.reader(f)
    y7 = [np.average(np.float_(row)) for row in reader]

with open("Official_both.csv") as f:
    reader = csv.reader(f)
    y8 = [np.average(np.float_(row)) for row in reader]

with open("Official_out.csv") as f:
    reader = csv.reader(f)
    y9 = [np.average(np.float_(row)) for row in reader]


print(x1)
print(y1)


plt.xlim(0, len(x3))
plt.ylim(80, 185.1)
plt.rcParams["font.size"] = 16

plt.plot(x1, y1, color='b')
plt.plot(x2, y2,  color='r')
plt.plot(x3, y3, color='g')
plt.plot(x1, y4, color='b')
plt.plot(x2, y5,  color='r')
plt.plot(x3, y6,  color='g')
plt.plot(x1, y7, label="MF", color='orchid')
plt.plot(x2, y8, label="Both", color='black')
plt.plot(x3, y9, label="Output", color='orange')

#pylab.subplots_adjust(right=0.7)
plt.fill_between(x1, y1, y4, color="lightblue", alpha=0.5)
plt.fill_between(x2, y2, y5, color="lightsalmon", alpha=0.5)
plt.fill_between(x3, y3, y6, color="lightgreen", alpha=0.5)
plt.title("ランダムな環境での評価値の範囲の推移" ,y=-0.30,fontname="MS Gothic")
plt.xlabel("評価回数",fontname="MS Gothic")
plt.ylabel("評価値",fontname="MS Gothic")
plt.rcParams['figure.dpi'] = 300

#plt.title("run10")
plt.legend(loc=4)


plt.show()
