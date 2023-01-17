import numpy as np
import time


ll = [3, 5, 2, 3, 1, 8, 7, 4, 6, 3, 2, 1, 10]
al = [2, 5, 1, 8, 7, 5, 4, 2, 1, 8, 9, 12, 13]
time_sta = time.time()
for i in range(100000):
      ll = np.array(ll)
      al = np.array(al)
      c = ll + al
time_end = time.time()

print(time_end - time_sta)
#0.015625