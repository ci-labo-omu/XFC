import math
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import japanize_matplotlib
import pylab
names = ['Near', 'Close', 'Remote']
"""
B1 = ctrl.Antecedent(np.arange(0, 40, 1), 'Temperature')
B1['Cold'] = fuzz.trimf(B1.universe, [-20, 0, 20])
B1['Moderate'] = fuzz.trimf(B1.universe, [0, 20, 40])
B1['Hot'] = fuzz.trimf(B1.universe, [20, 40, 60])"""
x = np.arange(0, 181, 1)
B1a = fuzz.trimf(x, [0, 0, 90])
B1b = fuzz.trimf(x, [0, 90, 180])
B1c = fuzz.trimf(x, [90, 180, 180])

fig = plt.figure()
ax = fig.subplots()
ax.plot(x, B1a, 'b', linewidth=1.5, label='Front')
ax.plot(x, B1b, 'g', linewidth=1.5, label='Side')
ax.plot(x, B1c, 'r', linewidth=1.5, label='Back')
ax.set_xlim(0.0, 180.1)
ax.set_ylim(0.0, 1.0)
plt.rcParams["font.family"] = "Times new roman"
plt.rcParams["font.size"] = 14
plt.rc('legend',loc='upper right',)
plt.rcParams["font.size"] = 16
ax.tick_params(labelsize=16, )
plt.xticks(np.arange(0, 181, step=45))
plt.ylabel("μ(x)", fontsize=20)
plt.xlabel("角度", fontsize=20, fontname='MS Gothic')
pylab.subplots_adjust(bottom=0.15)

ax.legend()
plt.savefig("anglesample_paper.png", format="png",dpi=300)
