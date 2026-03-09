import numpy as np

X, Y = [], []

for sim in range(1000):
    # 跑你现有 MATLAB 仿真（或复刻一份 Python 简化版）
    for k in range(T):
        phi = [
            np.linalg.norm(v_cv),
            np.linalg.norm(v_ctrv),
            np.trace(P_cv),
            np.trace(P_ctrv),
            abs(w_gt)
        ]
        label = [1, 0] if motion == "CV" else [0, 1]
        X.append(phi)
        Y.append(label)

np.save("X.npy", X)
np.save("Y.npy", Y)
