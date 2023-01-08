import math
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import skfuzzy.control as ctrl

names = ['Cold', 'Moderate', 'Hot']
"""
B1 = ctrl.Antecedent(np.arange(0, 40, 1), 'Temperature')
B1['Cold'] = fuzz.trimf(B1.universe, [-20, 0, 20])
B1['Moderate'] = fuzz.trimf(B1.universe, [0, 20, 40])
B1['Hot'] = fuzz.trimf(B1.universe, [20, 40, 60])"""
x = np.arange(0, 41, 1)
B1a = fuzz.trimf(x, [0, 0, 20])
B1b = fuzz.trimf(x, [0, 20, 40])
B1c = fuzz.trimf(x, [20, 40, 40])

fig = plt.figure()
ax = fig.subplots()
ax.plot(x, B1a, 'b', linewidth=1.5, label='Cold')
ax.plot(x, B1b, 'g', linewidth=1.5, label='Moderate')
ax.plot(x, B1c, 'r', linewidth=1.5, label='Hot')
ax.set_xlim(0.0, 40.1)
ax.set_ylim(0.0, 1.0)
ax.legend()
#plt.rc('legend', fontsize= 12, loc='upper right',)
plt.show()
plt.savefig("fuzzycenter.png", format="png")
