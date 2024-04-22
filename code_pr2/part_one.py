from preprocess import sync
from load_data import get_data

import autograd.numpy as np
from autograd import grad
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# ------------------- Data Set 20 -------------------------------------- #


data_encoder, _, data_imu, _ = get_data(20)

# Get encoder data
encoder_counts = data_encoder[0]
encoder_stamps = data_encoder[1]

# Get IMU data
imu_angular_velocity = data_imu[0]
imu_stamps = data_imu[2]

#Sync data for imu
new_imu_angular_velocity = np.zeros((3, len(encoder_stamps)))
new_imu_stamps = np.zeros(len(encoder_stamps))
j = 0
for i in range(len(encoder_stamps)):

    idx = np.argmin(np.abs(encoder_stamps[i] - imu_stamps))
    new_imu_angular_velocity[:, j] = imu_angular_velocity[:, idx]
    new_imu_stamps[j] = imu_stamps[idx]
    j += 1

#Predict robot motion via Euler - angle discretization
#Initialization
x_curr = np.zeros(3) #Might change this later
tau_curr = encoder_stamps[0] #Time difference
v_curr = 0.0022 / tau_curr #Linear velocity
omega_curr = np.zeros(3) #Current yaw angular velocity


x_next = np.zeros((3, len(encoder_stamps)))
for i in range(len(encoder_stamps) - 1):

    #Euler - angle discretization
    x_next[:, i] = x_curr + tau_curr*np.array([v_curr*np.cos(x_curr[2]), v_curr*np.sin(x_curr[2]), omega_curr[2]])

    #Update
    tau_curr = encoder_stamps[i + 1] - encoder_stamps[i]
    v_curr = ((((encoder_counts[:, i][0] + encoder_counts[:, i][1])/2)*0.0022) + (((encoder_counts[:, i][2] + encoder_counts[:, i][3])/2)*0.0022))/2
    omega_curr = np.array([0, 0, new_imu_angular_velocity[:, i][2]])
    x_curr = x_next[:, i]

plt.figure()
plt.plot(x_next[0, :], x_next[1, :], label = 'Data Set 20')
plt.title("Differential - drive robot trajectory for Data Set 20")

# ------------------- Data Set 21 -------------------------------------- #


data_encoder, _, data_imu, _ = get_data(21)

# Get encoder data
encoder_counts = data_encoder[0]
encoder_stamps = data_encoder[1]

# Get IMU data
imu_angular_velocity = data_imu[0]
imu_stamps = data_imu[2]

#Sync data for imu
new_imu_angular_velocity = np.zeros((3, len(encoder_stamps)))
new_imu_stamps = np.zeros(len(encoder_stamps))
j = 0
for i in range(len(encoder_stamps)):

    idx = np.argmin(np.abs(encoder_stamps[i] - imu_stamps))
    new_imu_angular_velocity[:, j] = imu_angular_velocity[:, idx]
    new_imu_stamps[j] = imu_stamps[idx]
    j += 1

#Predict robot motion via Euler - angle discretization
#Initialization
x_curr = np.zeros(3) #Might change this later
tau_curr = encoder_stamps[0] #Time difference
v_curr = 0.0022 / tau_curr #Linear velocity
omega_curr = np.zeros(3) #Current yaw angular velocity


x_next = np.zeros((3, len(encoder_stamps)))
for i in range(len(encoder_stamps) - 1):

    #Euler - angle discretization
    x_next[:, i] = x_curr + tau_curr*np.array([v_curr*np.cos(x_curr[2]), v_curr*np.sin(x_curr[2]), omega_curr[2]])

    #Update
    tau_curr = encoder_stamps[i + 1] - encoder_stamps[i]
    v_curr = ((((encoder_counts[:, i][0] + encoder_counts[:, i][1])/2)*0.0022) + (((encoder_counts[:, i][2] + encoder_counts[:, i][3])/2)*0.0022))/2
    omega_curr = np.array([0, 0, new_imu_angular_velocity[:, i][2]])
    x_curr = x_next[:, i]

plt.plot(x_next[0, :], x_next[1, :], label = 'Data Set 21')
plt.title("Differential - drive robot trajectory for Data Set 20 and 21")
plt.legend()
plt.show()



