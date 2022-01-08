import numpy as np

V = {'L1': 0.0, 'L2': 0.0}
V_new = V.copy()

for _ in range(100):
    V_new['L1'] = 0.5 * (-1 + 0.9 * V['L1']) + 0.5 * (1 + 0.9 * V['L2'])
    V_new['L2'] = 0.5 * (0 + 0.9 * V['L1']) + 0.5 * (-1 + 0.9 * V['L2'])
    V = V_new.copy()
    print(V)
