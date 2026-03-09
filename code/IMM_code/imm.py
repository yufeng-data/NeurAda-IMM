import numpy as np
import matplotlib.pyplot as plt

# CTRVKF.py
def CTRVKF(ctrv_parameter, z, x, ctrv_P, dt):
    ctrv_H = ctrv_parameter['ctrv_H']
    ctrv_Q = ctrv_parameter['ctrv_Q']
    ctrv_R = ctrv_parameter['ctrv_R']
    
    v_i = x[2]
    cos_i = x[3]
    sin_i = x[4]
    w_i = x[5]
    
    # 计算状态矩阵
    A_13 = 1/w_i * (sin_i*(np.cos(w_i*dt)-1) + cos_i*np.sin(w_i*dt))
    A_14 = v_i/w_i * np.sin(w_i*dt)
    A_15 = v_i/w_i * (np.cos(w_i*dt)-1)
    A_16 = (-v_i/w_i**2 * (sin_i*(np.cos(w_i*dt)-1) + cos_i*np.sin(w_i*dt)) 
            + v_i*dt/w_i * (-sin_i*np.sin(w_i*dt) + cos_i*np.cos(w_i*dt)))
    
    A_23 = 1/w_i * (cos_i*(1 - np.cos(w_i*dt)) + sin_i*np.sin(w_i*dt))
    A_25 = v_i/w_i * np.sin(w_i*dt)
    A_24 = -v_i/w_i * (np.cos(w_i*dt)-1)
    A_26 = (-v_i/w_i**2 * (cos_i*(1 - np.cos(w_i*dt)) + sin_i*np.sin(w_i*dt)) 
            + v_i*dt/w_i * (sin_i*np.cos(w_i*dt) + cos_i*np.sin(w_i*dt)))
    
    ctrv_A = np.array([
        [1, 0, A_13, A_14, A_15, A_16],
        [0, 1, A_23, A_24, A_25, A_26],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, np.cos(w_i*dt), -np.sin(w_i*dt), -cos_i*dt*np.sin(w_i*dt) - sin_i*dt*np.cos(w_i*dt)],
        [0, 0, 0, np.sin(w_i*dt), np.cos(w_i*dt), -sin_i*dt*np.sin(w_i*dt) + cos_i*dt*np.cos(w_i*dt)],
        [0, 0, 0, 0, 0, 1]
    ])
    
    x_pred = ctrv_A @ x
    ctrv_P = ctrv_A @ ctrv_P @ ctrv_A.T + ctrv_Q
    K = ctrv_P @ ctrv_H.T @ np.linalg.inv(ctrv_H @ ctrv_P @ ctrv_H.T + ctrv_R)
    x = x_pred + K @ (z - ctrv_H @ x_pred)
    ctrv_P = (np.eye(6) - K @ ctrv_H) @ ctrv_P
    
    # 规整化
    norm_val = np.linalg.norm(x[3:5])
    if norm_val > 0:
        x[3:5] = x[3:5] / norm_val
        x[2] = x[2] * norm_val
    
    return ctrv_A, x, ctrv_P, x_pred

# CVKF.py
def CVKF(cv_parameter, z, x, cv_P, dt):
    cv_Q = cv_parameter['cv_Q']
    cv_R = cv_parameter['cv_R']
    cv_H = cv_parameter['cv_H']
    
    v = x[2]
    cos_i = x[3]
    sin_i = x[4]
    
    cv_A = np.eye(6) + np.array([
        [0, 0, cos_i, v, 0, 0],
        [0, 0, sin_i, 0, v, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ]) * dt
    
    x_pred = cv_A @ x
    cv_P = cv_A @ cv_P @ cv_A.T + cv_Q
    K = cv_P @ cv_H.T @ np.linalg.inv(cv_H @ cv_P @ cv_H.T + cv_R)
    x = x_pred + K @ (z - cv_H @ x_pred)
    cv_P = (np.eye(6) - K @ cv_H) @ cv_P
    
    # 规整化
    norm_val = np.linalg.norm(x[3:5])
    if norm_val > 0:
        x[3:5] = x[3:5] / norm_val
        x[2] = x[2] * norm_val
    
    return cv_A, x, cv_P, x_pred

# IMMfilter.py
def IMMfilter(parameter, z, dt):
    P_model = parameter['P_model']
    mu_weight = parameter['mu_weight']
    model = parameter['model']
    
    c_bar = np.zeros(2)
    for i in range(2):
        c_bar[i] = np.sum(P_model[i, :] * mu_weight)
    
    mu = np.zeros((2, 2))
    for i in range(2):
        mu[i, :] = P_model[i, :] * mu_weight[i] / c_bar
    
    for i in range(2):
        model[i]['x_pre'] = model[i]['x'].copy()
        model[i]['P_pre'] = model[i]['P'].copy()
    
    for i in range(2):
        model[i]['x'] = np.zeros(6)
        for j in range(2):
            model[i]['x'] += mu[j, i] * model[j]['x_pre']
        norm_val = np.linalg.norm(model[i]['x'][3:5])
        if norm_val > 0:
            model[i]['x'][3:5] /= norm_val
    
    for i in range(2):
        model[i]['P'] = np.zeros((6, 6))
        for j in range(2):
            delta_x = model[i]['x'] - model[j]['x_pre']
            model[i]['P'] += mu[j, i] * (model[j]['P_pre'] + delta_x.reshape(-1, 1) @ delta_x.reshape(1, -1))
    
    # CVKF更新
    _, model[0]['x'], model[0]['P'], cv_x_pred = CVKF(parameter, z, model[0]['x'], model[0]['P'], dt)
    # CTRVKF更新
    _, model[1]['x'], model[1]['P'], ctrv_x_pred = CTRVKF(parameter, z, model[1]['x'], model[1]['P'], dt)
    
    model[0]['v'] = z - parameter['cv_H'] @ cv_x_pred
    model[1]['v'] = z - parameter['ctrv_H'] @ ctrv_x_pred
    
    model[0]['S'] = parameter['cv_H'] @ model[0]['P'] @ parameter['cv_H'].T + parameter['cv_R']
    model[1]['S'] = parameter['ctrv_H'] @ model[1]['P'] @ parameter['ctrv_H'].T + parameter['ctrv_R']
    
    Hat = np.zeros(2)
    Hat[0] = 1 / (np.sqrt(2 * np.pi) * np.sqrt(np.linalg.det(model[0]['S']))) * \
             np.exp(-0.5 * model[0]['v'].T @ np.linalg.inv(model[0]['S']) @ model[0]['v'])
    Hat[1] = 1 / (np.sqrt(2 * np.pi) * np.sqrt(np.linalg.det(model[1]['S']))) * \
             np.exp(-0.5 * model[1]['v'].T @ np.linalg.inv(model[1]['S']) @ model[1]['v'])
    
    c = np.sum(Hat * c_bar)
    mu_weight = (Hat * c_bar) / c
    
    x = mu_weight[0] * model[0]['x'] + mu_weight[1] * model[1]['x']
    P = (model[0]['P'] + (x - model[0]['x']).reshape(-1, 1) @ (x - model[0]['x']).reshape(1, -1)) * mu_weight[0] + \
        (model[1]['P'] + (x - model[1]['x']).reshape(-1, 1) @ (x - model[1]['x']).reshape(1, -1)) * mu_weight[1]
    
    parameter['mu_weight'] = mu_weight
    parameter['model'] = model
    
    return parameter, x, P

# 主程序
def main():
    np.random.seed(0)
    # 参数初始化
    t = np.arange(0.1, 20.1, 0.1)
    dt = 0.1
    video_save = False
    
    # 生成真实轨迹
    x_truth = []
    p = np.array([0, 0], dtype=np.float64)
    for i in range(200):
        if i < 50:
            v = np.array([5, 0])
        elif i < 150:
            r = 5
            v_mag = 10
            w = v_mag / r
            theta = (i - 50) * dt * w
            v = np.array([10 * np.cos(theta), 10 * np.sin(theta)])
        else:
            v = np.array([3, 5])
        p += v * dt
        x_truth.append(np.concatenate((p, v)))
    x_truth = np.array(x_truth).T
    
    # 生成观测数据
    z = x_truth[:2, :] + np.random.randn(2, 200) * 0.5
    
    # 初始化滤波器参数
    # CV参数
    cv_parameter = {
        'cv_H': np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]),
        'cv_Q': np.diag([0.1, 0.1, 1, 0.01, 0.01, 0]),
        'cv_R': np.eye(2) * 1
    }
    
    # CTRV参数
    ctrv_parameter = {
        'ctrv_H': np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]),
        'ctrv_Q': np.diag([0.1, 0.1, 1, 0.01, 0.01, 1]),
        'ctrv_R': np.eye(2) * 1.2
    }
    
    # IMM参数
    IMM_parameter = {
        'cv_H': cv_parameter['cv_H'],
        'cv_Q': cv_parameter['cv_Q'],
        'cv_R': cv_parameter['cv_R'],
        'ctrv_H': ctrv_parameter['ctrv_H'],
        'ctrv_Q': ctrv_parameter['ctrv_Q'],
        'ctrv_R': ctrv_parameter['ctrv_R'],
        'P_model': np.array([[0.96, 0.04], [0.04, 0.96]]),
        'mu_weight': np.array([0.7, 0.3]),
        'model': [
            {'x': np.array([0, 0, 0, 1, 0, 0]), 'P': np.eye(6)},
            {'x': np.array([0, 0, 0, 1, 0, 0.1]), 'P': np.eye(6)}
        ]
    }
    
    # 初始化滤波器状态
    cv_x_estimation = np.zeros((6, 201))
    cv_x_estimation[3, 0] = 1
    cv_P = np.eye(6)
    
    ctrv_x_estimation = np.zeros((6, 201))
    ctrv_x_estimation[4, 0] = 1
    ctrv_x_estimation[5, 0] = 0.1
    ctrv_P = np.diag([1, 1, 1, 1, 1, 1])
    
    IMM_x_estimation = np.zeros((6, 201))
    IMM_x_estimation[4, 0] = 1
    IMM_x_estimation[5, 0] = 0.1
    IMM_save = {'mu': []}
    
    # 主循环
    for i in range(200):
        # 预测步骤（略，参考Matlab代码逻辑）
        # 更新步骤
        # CV更新
        _, cv_x_estimation[:, i+1], cv_P, _ = CVKF(cv_parameter, z[:, i], cv_x_estimation[:, i], cv_P, dt)
        # CTRV更新
        _, ctrv_x_estimation[:, i+1], ctrv_P, _ = CTRVKF(ctrv_parameter, z[:, i], ctrv_x_estimation[:, i], ctrv_P, dt)
        # IMM更新
        IMM_parameter, IMM_x_estimation[:, i+1], _ = IMMfilter(IMM_parameter, z[:, i], dt)
        IMM_save['mu'].append(IMM_parameter['mu_weight'])
    
    # 绘图
    plt.figure(figsize=(12, 8))
    plt.plot(x_truth[0, :], x_truth[1, :], 'k-', label='Truth Trajectory')
    plt.plot(z[0, :], z[1, :], '.', markersize=5, label='Measurements')
    plt.plot(cv_x_estimation[0, :200], cv_x_estimation[1, :200], 'b-', label='CV-KF')
    plt.plot(ctrv_x_estimation[0, :200], ctrv_x_estimation[1, :200], 'g-', label='CTRV-KF')
    plt.plot(IMM_x_estimation[0, :200], IMM_x_estimation[1, :200], 'r-', label='IMM')
    plt.legend()
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.title('Trajectory Comparison')
    plt.show()
    
    # 误差图
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, cv_x_estimation[0, 1:201] - x_truth[0, :], 'b', label='CV-KF')
    plt.plot(t, ctrv_x_estimation[0, 1:201] - x_truth[0, :], 'g', label='CTRV-KF')
    plt.plot(t, IMM_x_estimation[0, 1:201] - x_truth[0, :], 'r', label='IMM')
    plt.ylabel('X Error (m)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, cv_x_estimation[1, 1:201] - x_truth[1, :], 'b', label='CV-KF')
    plt.plot(t, ctrv_x_estimation[1, 1:201] - x_truth[1, :], 'g', label='CTRV-KF')
    plt.plot(t, IMM_x_estimation[1, 1:201] - x_truth[1, :], 'r', label='IMM')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Error (m)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # IMM权重图
    plt.figure(figsize=(12, 6))
    mu_data = np.array(IMM_save['mu'])
    plt.plot(t, mu_data[:, 0], label='IMM-CV Weight')
    plt.plot(t, mu_data[:, 1], label='IMM-CTRV Weight')
    plt.xlabel('Time (s)')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()