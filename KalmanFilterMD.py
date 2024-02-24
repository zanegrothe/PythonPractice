# Kalman Filter calculation practice (Multi-Dimensional)

import numpy as np
import matplotlib.pyplot as plt

# Initial Conditions
x = 4000  # m
vx = 280  # m/s
X = ([x], [vx])
ax = 2  # m/s^2
dt = 1  # s

# Errors
Eest = np.array([[20], [5]])  # estimate error
Emea = np.array([[25], [6]])  # measurement error
P = np.diag(np.diag(np.dot(Eest, Eest.transpose())))

# Adaptation Matrices
A = np.array([[1, dt], [0, 1]])
B = np.array([[0.5 * dt ** 2], [dt]])
C = np.identity(len(Emea))

# Transformation Matrix
H = np.identity(len(Emea))

# Noise
w = 0  # Predicted State noise
Q = 0  # Process noise
R = np.diag(np.diag(np.dot(Emea, Emea.transpose())))  # Sensor noise
z = 0  # Measurement noise

# Measurements
MEAx = np.array([4260, 4550, 4860, 5110])  # x position
MEAvx = np.array([282, 285, 286, 290])  # x velocity
MEA = np.array(list(zip(MEAx, MEAvx)))

# Calculations
time = np.array([])
errorMea = np.array([])
stateEst = np.array([])
kalmanGain = np.array([])
errorEst = np.array([])
t = 0

while t < len(MEAx):
    X = np.dot(A, X) + np.dot(B, ax) + w
    P = np.diag(np.diag(np.dot(np.dot(A, P), A.transpose()) + Q))
    KG = np.divide(np.dot(P, H.transpose()), np.dot(np.dot(H, P), H.transpose()) + R,
                   out=np.zeros_like(np.dot(P, H.transpose())),
                   where=np.dot(np.dot(H, P), H.transpose()) + R != 0)
    mea = np.array([[MEA[t][0]], [MEA[t][1]]])
    Y = np.dot(C, mea) + z
    Y = np.array([Y[0], Y[1]])
    X = X + np.dot(KG, (Y - np.dot(H, X)))
    P = np.dot((np.identity(len(Emea)) - np.dot(KG, H)), P)
    time = np.append(time, t)
    errorMea = np.append(errorMea, Emea)
    stateEst = np.append(stateEst, X)
    kalmanGain = np.append(kalmanGain, np.diag(KG))
    errorEst = np.append(errorEst, np.diag(P))
    t += 1


# Data
outputData = np.transpose(np.array([time,
                                    MEAx.tolist(), MEAvx.tolist(),
                                    errorMea[0::2], errorMea[1::2],
                                    stateEst[0::2], stateEst[1::2],
                                    kalmanGain[0::2], kalmanGain[1::2],
                                    errorEst[0::2], errorEst[1::2]]))
print('Time__Measurement(x,v)__Measurement Error(x,v)__Estimate(x,v)__Kalman Gain(x,v)__Error Estimate(x,v)')
np.set_printoptions(suppress=True)
print(outputData.round(3))

# Plots

# Measurements vs Estimates
plt.figure(1)
plt.subplot(2, 1, 1)
plt.scatter(time, MEAx, c='r', label='Measurements')
plt.plot(time, stateEst[0::2], 'g', label='Estimation')
plt.xlabel('Measurement')
plt.xticks(time)
plt.ylabel('Position')
plt.title('Kalman Filter Estimates for Temperature')
plt.legend()
plt.tight_layout()

plt.subplot(2, 1, 2)
plt.scatter(time, MEAvx, c='r', label='Measurements')
plt.plot(time, stateEst[1::2], 'g', label='Estimation')
plt.xlabel('Measurement')
plt.xticks(time)
plt.ylabel('Velocity')
plt.legend()
plt.tight_layout()


# Errors
plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(time, errorEst[0::2], 'b')
plt.xlabel('Measurement')
plt.ylabel('Position Error')
plt.title('Kalman Filter Error for Estimates')
plt.tight_layout()

plt.subplot(2, 1, 2)
plt.plot(time, errorEst[1::2], 'b')
plt.xlabel('Measurement')
plt.ylabel('Velocity Error')
plt.tight_layout()


# Kalman Gains
plt.figure(3)
plt.subplot(2, 1, 1)
plt.plot(time, kalmanGain[0::2], 'm')
plt.xlabel('Measurement')
plt.ylabel('Position Kalman Gain')
plt.title('Kalman Filter Kalman Gain')
plt.tight_layout()

plt.subplot(2, 1, 2)
plt.plot(time, kalmanGain[1::2], 'm')
plt.xlabel('Measurement')
plt.ylabel('Velocity Kalman Gain')
plt.tight_layout()
plt.show()
