import skfuzzy as fuzz
import skfuzzy.control as ctrl
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools

N_pop = 30
N_rep = 10
count_max = 10000
p = 0.9
# 分割数
K = 3

df1 = pd.read_csv('data/kadai3_pattern1.txt', header=None, skiprows=[0])
df2 = pd.read_csv('data/kadai3_pattern2.txt', header=None, skiprows=[0])
df1.columns = ["x", "y", "label"]
df2.columns = ["x", "y", "label"]

Ruledict = {
    0: "x1 is small and x2 is small",
    1: "x1 is small and x2 is medium",
    2: "x1 is small and x2 is large",
    3: "x1 is small and x2 is don't care",
    4: "x1 is medium and x2 is small",
    5: "x1 is medium and x2 is medium",
    6: "x1 is medium and x2 is large",
    7: "x1 is medium and x2 is don't care",
    8: "x1 is large and x2 is small",
    9: "x1 is large and x2 is medium",
    10: "x1 is large and x2 is large",
    11: "x1 is large and x2 is don't care",
    12: "x1 is don't care and x2 is small",
    13: "x1 is don't care and x2 is medium",
    14: "x1 is don't care and x2 is large",
    15: "x1 is don't care and x2 is don't care"

}

x_array = df1.values
x_values = np.delete(x_array, 2, 1)


def membership(x):
    M = np.ones((len(x), len(x[0]) - 1), dtype=float)
    for i in range(len(x)):
        # iじゃなくてlabelをうまく使いたい
        for j in range(len(x[0]) - 1):
            M[i][j] = (max(0, (1 - np.abs(x[i][2] / 2 - x[i][j]) * 2)))
    return M


def membership2(x, k):
    # q1が0から始まるからk=k-1
    if k == 3: return 1.0
    b = 1 / (K - 1)
    a = k / (K - 1)

    return (max(0, (1 - np.abs(a - x) / b)))


M = []
C_q = []
# 信頼度
c = []


def fit(x_array, i):
    q1 = i // 4
    q2 = i % 4

    m = np.array([membership2(x[0], q1) * membership2(x[1], q2) for x in x_array])
    return m


def rule_change(C_q, CF_q):
    return [b for b in C_q if CF_q[b] > 0]


def rule_count(C_q, CF_q):
    count = 2 * len(CF_q[CF_q > 0])
    index = np.where(CF_q > 0)
    for k in index[0]:
        if k % 4 == 3: count -= 1
        if k // 4 == 3: count -= 1

    return count


for i in range(16):
    m = fit(x_array, i)
    m_sum = np.array([0.0, 0.0, 0.0])
    for l in range(3):
        m_sum[l] = (np.sum(m[x_array[:, 2] == l]))
    c.append(m_sum / np.sum(m_sum))

# 信頼度
# print(c)

# 結論部
C_q = np.argmax(c, axis=1)

print(f"結論部:{C_q}")

CF_q = 2 * np.max(c, axis=1) - np.sum(c, axis=1)
rules = rule_change(range(16), CF_q)

pro = np.full((16, len(x_array)), -1.0)
for i in range(16):
    m = fit(x_array, i)
    pro[i] = m * CF_q[i]
R_w = np.argmax(pro, axis=0)
fit_x = C_q[R_w]
fit_x = np.count_nonzero(fit_x == x_array[:, 2]) / len(x_array)
print(f"識別率:{fit_x}")

for i in range(16):
    print(f"集合{i + 1:2d} : {Ruledict[i]} then Class{C_q[i] + 1}")
print(f"重み：\n{CF_q}")
print(f"ルール数:{len(rules)}")
print(f"総ルール長：{rule_count(rules, CF_q)}")

print(f"得られた識別器:")
for rule in rules:
    print(f"{Ruledict[rule]} then Class{C_q[rule] + 1}")


# グラフ描画
