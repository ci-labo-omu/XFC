import math
import numpy as np

a = np.array([2, 1])
b = np.array([6, 5])
ang = a - b
ca = (3, 4)
cb = (2, 8)
print(math.degrees(np.arctan2(ang[1], ang[0])))
print(math.dist(ca, cb))