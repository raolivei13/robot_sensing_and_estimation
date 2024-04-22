from load_data import get_data
from icp_warm_up import icp_file
from pr2_utils import bresenham2D
from scipy.special import expit
#from icp_warm_up import utils


# Modify this for data set number
dataset = 21


import autograd.numpy as np
import transforms3d as t3d

from autograd import grad
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')





def convert_to_rect(ranges, ang_incr, ang_range_min):

    # ranges is of dim N by 1
    rect_measure = np.zeros((len(ranges), 3))
    incr = 0
    for i in range(len(ranges)):

        rect_measure[i, :] = np.array([ranges[i] * np.cos(ang_range_min + incr), ranges[i] * np.sin(ang_range_min + incr), 514.35 / 1000])

        incr = incr + ang_incr

    return rect_measure # Dim N by 3

def create_pose(R, p):

    T = np.zeros((4, 4))
    # R is 3x3
    # p is 3x1

    T[0:3, 0:3] = R
    T[0, 3] = p[0]
    T[1, 3] = p[1]
    T[2, 3] = p[2]
    T[3, 3] = 1

    return T

# Create a Rotation matrix along the z - axis
def create_rotation_along_z(theta):

    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])


def create_rotation_along_y(theta):

    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])






#Gets encoder, IMU and Lidar data
data_encoder, data_hok, data_imu, _ = get_data(dataset)


# Get encoder data
encoder_counts = data_encoder[0]
encoder_stamps = data_encoder[1]

# Get IMU data
imu_angular_velocity = data_imu[0]
imu_stamps = data_imu[2]

# Get LiDar data
lidar_angle_min = data_hok[0]  # start angle of the scan [rad]
lidar_angle_max = data_hok[1]  # end angle of the scan [rad]
lidar_angle_increment = data_hok[2]  # angular distance between measurements [rad]
lidar_range_min = data_hok[3]  # minimum range value [m]
lidar_range_max = data_hok[4]  # maximum range value [m]
lidar_ranges = data_hok[5]  # range data [m] (Note: values < range_min or > range_max should be discarded)

# Discard and range_min and range_max
# TO DO

lidar_stamps = data_hok[6]  # acquisition times of the lidar scans


#Sync data for imu
new_imu_angular_velocity = np.zeros((3, len(encoder_stamps)))
new_imu_stamps = np.zeros(len(encoder_stamps))

#Sync data for LiDar
new_lidar_ranges = np.zeros((lidar_ranges.shape[0], len(encoder_stamps)))
new_lidar_stamps = np.zeros(len(encoder_stamps))

print("Data imported")

print("Starting Sync")
j = 0
for i in range(len(encoder_stamps)):

    #Sync IMU
    idx_imu = np.argmin(np.abs(encoder_stamps[i] - imu_stamps))
    new_imu_angular_velocity[:, j] = imu_angular_velocity[:, idx_imu]
    new_imu_stamps[j] = imu_stamps[idx_imu]

    #Sync LiDar
    idx_hok = np.argmin(np.abs(encoder_stamps[i] - lidar_stamps))
    new_lidar_ranges[:, j] = lidar_ranges[:, idx_hok]
    new_lidar_stamps[j] = lidar_stamps[idx_hok]
    j += 1

print("Sync ended")

# Robot odometry information
x_curr = np.zeros(3) #Might change this later
tau_curr = encoder_stamps[0] #Time difference
v_curr = 0.0022 / tau_curr #Linear velocity
omega_curr = np.zeros(3) #Current yaw angular velocity


print("Start robot odometry")
# ------------------ Get Robot Odometry ------------------------------ #
x_next = np.zeros((3, len(encoder_stamps)))
for i in range(len(encoder_stamps) - 1):

    #Euler - angle discretization
    x_next[:, i] = x_curr + tau_curr*np.array([v_curr*np.cos(x_curr[2]), v_curr*np.sin(x_curr[2]), omega_curr[2]])

    #Update
    tau_curr = encoder_stamps[i + 1] - encoder_stamps[i]
    v_curr = ((((encoder_counts[:, i][0] + encoder_counts[:, i][1])/2)*0.0022) + (((encoder_counts[:, i][2] + encoder_counts[:, i][3])/2)*0.0022))/2
    omega_curr = np.array([0, 0, new_imu_angular_velocity[:, i][2]])
    x_curr = x_next[:, i]

# ------------------------------------------------------------------- #

print("End robot odometry")


# Use robot odometry at each step to initialize ICP
# Outer for loop that iterates over the time stamps

print("Start Scan Matching")

x_opt_next = np.zeros((3, len(encoder_stamps))) #Define the empty optimized trajectory
#R_opt_curr = np.eye(3)
#p_opt_curr = np.zeros(3)
#T_opt_curr = np.zeros((4, 4))

T_opt_curr = create_pose(create_rotation_along_z(x_next[:, 0][2]), np.array([x_next[:, 0][1], x_next[:, 0][2], 127 / 1000]))

shift_to_robot_pos = np.array([0, 149.165/1000, 0]) # Position of Lidar wrt to robot geom center
R_Lidar = create_rotation_along_z(3 * np.pi / 2)
for i in range(len(encoder_stamps) - 1):


    # Range measurements over one scan at time step i
    lidar_measure_t = new_lidar_ranges[:, i]# Dim 1081 x 1
    lidar_measure_t_1 = new_lidar_ranges[:, i + 1] # Dim 1081 x 1

    #lidar_measure_rect_t = np.zeros((3, lidar_measure_t.shape[0]))
    #lidar_measure_rect_t_1 = np.zeros((3, lidar_measure_t_1.shape[0]))

    # LiDar Scans with body frame coordinates
    lidar_measure_rect_t = np.transpose(convert_to_rect(lidar_measure_t, lidar_angle_increment[0][0], lidar_angle_min)) + shift_to_robot_pos.reshape(3, 1) # This is the Lidar scan at time t
    lidar_measure_rect_t_1 = np.transpose(convert_to_rect(lidar_measure_t_1, lidar_angle_increment[0][0], lidar_angle_min)) + shift_to_robot_pos.reshape(3, 1) # This is the Lidar scan at time t + 1






    # Get Rotation matrix of robot at time t and at time t + 1

    # Translation at time t + 1 i.e position of robot at time t + 1
    p_t1 = np.array([x_next[:, i + 1][0], x_next[:, i + 1][1], 127 / 1000])
    # Translation at time t i.e position of robot at time t
    p_t = np.array([x_next[:, i][0], x_next[:, i][1], 127 / 1000])

    # pt cloud at time t + 1 with coordinates wrt the body frame
    target = lidar_measure_rect_t_1
    # pt cloud at time t with coordinates wrt the body frame
    source = lidar_measure_rect_t

    # Define the centroids of the two pt clouds
    #cen_target = np.mean(target, axis = 1)
    #cen_source = np.mean(source, axis = 1)

    # ------------------ Initialization with Robot odometry ----------------------- #
    R_curr = np.linalg.inv(create_rotation_along_z(x_next[:, i][2])) @ create_rotation_along_z(x_next[:, i + 1][2])
    p_curr = p_t1.reshape(3, 1) - p_t.reshape(3, 1)
    #print(np.linalg.det(R_curr))
    # ------------------ Initialization with Robot odometry ----------------------- #


    # 10 num of iterations for icp
    # Get the relative pose between the target and the source , i.e t_T_t+1

    #utils.visualize_icp_result(source, target, np.eye(4))

    t_T_t1, _ = icp_file.icp(50, source.T, target.T, R_curr, p_curr) # Relative pose between the two scan matches

    #utils.visualize_icp_result(source, target, t_T_t1)

    #if j == 3:
        #break

    # Flip by pi the poses here
    new_ang = -1 * t3d.euler.mat2euler(t_T_t1[0:3, 0:3])[2]
    t_T_t1[0:3, 0:3] = create_rotation_along_z(new_ang)

    # Estimate the robot pose at time t + 1 T_t+1
    T_t1 = np.matmul(T_opt_curr, t_T_t1)


    # Updates
    x_opt_next[:, i + 1][0] = T_t1[0, 3]
    x_opt_next[:, i + 1][1] = T_t1[1, 3]
    # rot = Rotation.from_matrix(T_t1[0:3, 0:3])
    # yaw, _, _ = rot.as_euler('zyx')
    x_opt_next[:, i + 1][2] = -1 * t3d.euler.mat2euler(T_t1[0:3, 0:3])[2]

    T_opt_curr = T_t1


    print("itertion: ", i)



print("Scan Matching Ended")

np.save('opt_icp_traj.npy', x_opt_next) # Save optimized trajectory

# Plot
# plt.figure()
#
# plt.plot(x_next[2, :], label = 'NOT optimized - Angle -  Data Set 20')
# plt.plot(x_opt_next[2, :], label = 'OPTIMIZED - Angle - Data Set 20')
# plt.legend()
# plt.show()



# plt.figure()
# plt.plot(x_next[0, :], x_next[1, :], label = 'NOT optimized - Data Set 20')
# plt.title("Differential - drive robot motion model trajectory for Data Set 20")
#
# plt.plot(x_opt_next[0, :], x_opt_next[1, :], label = 'OPTIMIZED - Data Set 20')
# plt.title("Differential - drive robot OPTIMIZED motion model trajectory for Data Set 20")
# plt.legend()
# plt.show()



#Define Grid size
# n1 = 800 # row dim
# n2 = 800 # col dim
# grid = np.ones((n1, n2))
# res = 0.05 # Resolution
# grid_center_x = int(grid.shape[0]/2 - 1)
# grid_center_y = int(grid.shape[1]/2 - 1)
# #
# # # Get first LiDar scan measurement
# # lidar_measure_first = np.delete(new_lidar_ranges[:, 0], [np.argmin(new_lidar_ranges[:, 0]), np.argmax(new_lidar_ranges[:, 0])])  # Dim 1081 x 1
# # lidar_measure_rect_first = convert_to_rect(lidar_measure_first, lidar_angle_increment[0][0], lidar_angle_min) # This is N by 3
# #
# #
# # # Starting point of LiDar in pixel frame
# # lidar_vec_start = np.array([0, 149.165 / 1000, 0]) / res
# #
# # # Lidar end points convert to World Frame
# # R = np.array([[np.cos(x_next[:, 0][2]), -np.sin(x_next[:, 0][2]), 0],
# #               [np.sin(x_next[:, 0][2]), np.cos(x_next[:, 0][2]), 0],
# #               [0, 0, 1]])
# # p = np.array([x_next[:, 0][0], x_next[:, 0][1], 0])
# # lidar_measure_rect_first_W = np.matmul(R, lidar_measure_rect_first.T) + p.reshape((3, 1))
# #
# # for i in range(lidar_measure_rect_first.shape[0]):
# #     filled_pixels = bresenham2D(lidar_vec_start[0], lidar_vec_start[1], lidar_measure_rect_first_W[:, i][0] / res, lidar_measure_rect_first_W[:, i][1] / res)
# #     #grid[grid_center_x + filled_pixels[0].astype(int), grid_center_y + filled_pixels[1].astype(int)] = 0
# #     # Decrease / Increase Log odds
# #     grid[grid_center_x + filled_pixels[0, :-1].astype(int), grid_center_y + filled_pixels[1, :-1].astype(int)] -= np.log(4)
# #     grid[grid_center_x + filled_pixels[0, -1].astype(int), grid_center_y + filled_pixels[1, -1].astype(int)] += np.log(4)
# #
# # occ_grid_sigmoid = expit(grid)
# # plt.figure()
# # plt.imshow(np.flip(occ_grid_sigmoid, axis=1).T)
# # plt.show()
# # plt.figure()
# # plt.imshow(grid, cmap='viridis', interpolation='nearest')
# # plt.title("Occupancy Grid for first LiDar Scan")
# # plt.show()
#
# # ----------------- Create occupancy Grid with the entire LiDar Scans --------------------------------- #
#
# print("Start Occupancy Grid")
# # Create an occupancy grid with the Odometry data
# # Iterate over the entire the time stamps
# for i in range(len(lidar_stamps)):
#
#     lidar_measure = np.delete(new_lidar_ranges[:, i], [np.argmin(new_lidar_ranges[:, i]), np.argmax(new_lidar_ranges[:, i])])  # Dim 1081 x 1
#     lidar_measure_rect = convert_to_rect(lidar_measure, lidar_angle_increment[0][0], lidar_angle_min)  # This is N by 3
#
#     # Starting point of LiDar in pixel frame
#     lidar_vec_start = np.array([0, 149.165 / 1000, 0]) / res
#
#     # Lidar end points convert to World Frame
#     R = np.array([[np.cos(x_opt_next[:, i][2]), -np.sin(x_opt_next[:, i][2]), 0],
#                   [np.sin(x_opt_next[:, i][2]), np.cos(x_opt_next[:, i][2]), 0],
#                   [0, 0, 1]])
#     p = np.array([x_opt_next[:, i][0], x_opt_next[:, i][1], 0])
#     lidar_measure_rect_W = np.matmul(R, lidar_measure_rect.T) + p.reshape((3, 1))
#
#     for j in range(lidar_measure_rect.shape[0]):
#         filled_pixels = bresenham2D(lidar_vec_start[0], lidar_vec_start[1], lidar_measure_rect_W[:, j][0] / res,
#                                     lidar_measure_rect_W[:, j][1] / res)
#         # grid[grid_center_x + filled_pixels[0].astype(int), grid_center_y + filled_pixels[1].astype(int)] = 0
#         # Decrease / Increase Log odds
#         grid[grid_center_x + filled_pixels[0, :-1].astype(int), grid_center_y + filled_pixels[1, :-1].astype(
#             int)] -= np.log(4)
#         grid[grid_center_x + filled_pixels[0, -1].astype(int), grid_center_y + filled_pixels[1, -1].astype(
#             int)] += np.log(4)
#
#     print("Iteration :", i)
#
#
# print("End Occupancy Grid")
# occ_grid_sigmoid = expit(grid)
# plt.figure()
# plt.imshow(np.flip(occ_grid_sigmoid, axis=1).T)
# # plt.plot(x_next[0, :], x_next[1, :])
# plt.show()


