# Kalman Filter calculation practice (1 Dimensional)

import numpy as np
import random
import matplotlib.pyplot as plt

trueTemp = 72

# Initial Estimate
EST = 68

# Errors
Eest = 2  # estimate error
Emea = 4  # measurement error

# Measurements
# TempMEA = [75, 71, 70, 74]

TempMEA = []
ms = 500  # total measurements
for i in range(ms):
    # TempMEA.append(random.randint(trueTemp - Emea, trueTemp + Emea))  # integers
    TempMEA.append(((trueTemp + Emea) - (trueTemp - Emea)) * random.random() + (trueTemp - Emea))  # floating

# Calculations
time = []
temp = []
errorMea = []
TempEST = []
kalmanGain = []
errorEst = []
t = 1
for T in TempMEA:
    KG = Eest / (Eest + Emea)  # Kalman Gain
    EST = EST + KG*(T - EST)  # Current Estimate
    Eest = (Emea * Eest) / (Emea + Eest)  # Current Error in Estimate
    time.append(t)
    temp.append(trueTemp)
    errorMea.append(Emea)
    TempEST.append(EST)
    kalmanGain.append(KG)
    errorEst.append(Eest)
    t += 1

# Data
outputData = np.transpose(np.array([time, TempMEA, errorMea, TempEST, kalmanGain, errorEst]))
print('Time__Temp Measurement__Measurement Error__Temp Estimate__Kalman Gain__Error Estimate')
np.set_printoptions(suppress=True)
print(outputData.round(3))

# Plots
plt.figure(1)
plt.plot(time, temp, 'k', label='True Temperature')
plt.scatter(time, TempMEA, s=5, c='r', label='Measurements')
plt.plot(time, TempEST, 'g', label='Estimation')
plt.xlabel('Measurement')
# plt.xticks(time)
plt.ylabel('Temperature')
plt.title('Kalman Filter Estimates for Temperature')
plt.legend()
plt.tight_layout()

plt.figure(2)
plt.plot(time, errorEst, 'b')
plt.xlabel('Measurement')
plt.ylabel('Error')
plt.title('Kalman Filter Error for Estimates')
plt.tight_layout()

plt.figure(3)
plt.plot(time, kalmanGain, 'm')
plt.xlabel('Measurement')
plt.ylabel('Kalman Gain')
plt.title('Kalman Filter Kalman Gain')
plt.tight_layout()
plt.show()
