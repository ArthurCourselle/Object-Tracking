import numpy as np


class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.dt = dt
        self.u = np.array([u_x, u_y])
        self.std_acc = std_acc
        self.x_std_meas = x_std_meas
        self.y_std_meas = y_std_meas
        self.x = [0, 0, 0, 0]
        self.A = np.array(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        self.B = np.array(
            [[0.5 * self.dt**2, 0], [0, 0.5 * self.dt**2], [self.dt, 0], [0, self.dt]]
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.P = np.eye(self.A.shape[1])
        self.Q = (
            np.array(
                [
                    [self.dt**4 / 4, 0, self.dt**3 / 2, 0],
                    [0, self.dt**4 / 4, 0, self.dt**3 / 2],
                    [self.dt**3 / 2, 0, self.dt**2, 0],
                    [0, self.dt**3 / 2, 0, self.dt**2],
                ]
            )
            * self.std_acc**2
        )
        self.R = np.array([[self.x_std_meas**2, 0], [0, self.y_std_meas**2]])

    def predict(self):
        X_new = self.A @ self.x + self.B @ self.u
        P_new = self.A @ self.P @ self.A.T + self.Q
        self.x = X_new
        self.P = P_new
        return X_new

    def update(self, measurement):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        X_new = self.x + K @ (measurement - (self.H @ self.x))
        I = np.eye(self.H.shape[1])
        P_new = (I - K @ self.H) @ self.P
        self.x = X_new
        self.P = P_new
        return X_new
