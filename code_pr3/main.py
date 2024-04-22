
import numpy as np
import time
from pr3_utils import *
from scipy.linalg import expm
from scipy.linalg import block_diag

# --------- Functions ----------- #


def op(vec):

	# vec in homogeneous coordinates 4 by 1
	op = np.zeros((4, 6))
	op[0:3, 0:3] = np.eye(3)
	op[0:3, 3:] = -1 * create_hat_map(vec)

	return op


def create_hat_map(vec):

	# Given a vector in R^3 compute the hat map

	hat_vec = np.zeros((3, 3))

	hat_vec[0, 1] = -1 * vec[2]
	hat_vec[0, 2] = vec[1]
	hat_vec[1, 0] = vec[2]
	hat_vec[1, 2] = -1 * vec[0]
	hat_vec[2, 0] = -1 * vec[1]
	hat_vec[2, 1] = vec[0]

	return hat_vec

def create_adj_from_hat_map(hat_map):

	# hat_map is 4 by 4
	# creates an adjoint of the hat map in 6 by 6

	adj_hat_map = np.zeros((6, 6))

	adj_hat_map[0:2, 0:2] = hat_map[0:2, 0:2]
	adj_hat_map[0:2, 3:5] = create_hat_map(hat_map[0:2, 3])
	adj_hat_map[3:5, 3:5] = hat_map[0:2, 0:2]

	return  adj_hat_map

# --------- Functions ----------- #



# Load the measurements
filename = "data/10.npz" # ChangeDataSet
t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename)

print("Data imported")
#print(t[0])

# (a) IMU Localization via EKF Prediction

# Set Gaussian noise for motion model
pert_w = 1e-3
pert_v = 1e-4
W = pert_w * np.eye(6)

world_T_imu = np.zeros((4, 4, t.shape[1]))
world_T_imu[:, :, 0] = np.eye(4) # Specify Prior pose
covs_over_time = np.zeros((6, 6, t.shape[1]))
cov = np.diag([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]) # Set intial covariance
covs_over_time[:, :, 0] = cov


# ------------- EKF part 1 --------------------- #


# state_pos_angle = np.zeros((t.shape[1], 6))
# for i in range(t.shape[1] - 1):
# 	state_pos_angle[i, :] = np.array([(t[0][i + 1] - t[0][i]) * linear_velocity[:, i][0], (t[0][i + 1] - t[0][i]) * linear_velocity[:, i][1], (t[0][i + 1] - t[0][i]) * linear_velocity[:, i][2], (t[0][i + 1] - t[0][i]) * angular_velocity[:, i][0], (t[0][i + 1] - t[0][i]) * angular_velocity[:, i][1], (t[0][i + 1] - t[0][i]) * angular_velocity[:, i][2]])
#
#
# state_pos_angle_pose = twist2pose(axangle2twist(state_pos_angle)) # Regular poses i.e exponentiated twists
# state_pos_angle_adj_pose = twist2pose(-1 * axangle2adtwist(state_pos_angle)) # Adjoint of pose
# current_world_T_imu = world_T_imu[:, :, 0]
# current_cov = covs_over_time[:, :, 0]
# for i in range(t.shape[1] - 1):
#
#
# 	# Implement prediction step
# 	world_T_imu[:, :, i + 1] = current_world_T_imu @ state_pos_angle_pose[i, :, :]
# 	covs_over_time[:, :, i + 1] = state_pos_angle_adj_pose[i, :, :] @ current_cov @ np.transpose(state_pos_angle_adj_pose[i, :, :]) + W
#
# 	current_world_T_imu = world_T_imu[:, :, i + 1]
# 	current_cov = covs_over_time[:, :, i + 1]


# visualize_trajectory_2d(world_T_imu, show_ori = True)


print("Done with update")

# ------------- EKF part 1 --------------------- #








# (b) Landmark Mapping via EKF Update
# features vector is 4 by N by T, each 4 by N corresponds to visual features observed at time t
# Start removing outliers in the features

rem_outliers = True
skip = 20  # Get every 20th observation
eps = 1e-4 # Small perturbation in covariance matrix
outlier_dist = 180 # Can change this later
num_outliers = 0 # Number of outliers
outlier_t = np.linalg.norm(np.array([outlier_dist, outlier_dist])) # Define the threshold
features = features[:, ::skip] # Down sample the amount of features to deal with
mu_start = np.zeros((4, features.shape[1])) # Initialize the mean position of landmark locations
cov_start = eps * np.eye(3*features.shape[1])

# Full covariance matrix for SLAM
covar_slam = eps * np.eye(3*features.shape[1]+6)
world_T_imu_slam = np.zeros((4, 4, t.shape[1]))
world_T_imu_slam[:, :, 0] = np.eye(4)




#Define important matrices
cam_R_reg = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
cam_T_imu = np.linalg.inv(imu_T_cam)

K_s = np.zeros((4, 4))
K_s[0:2, 0:3] = K[0:2, 0:3]
K_s[2:, 0:3] = K[0:2, 0:3]
K_s[2, 3] = -1 * K[0, 0] * b

P = np.zeros((3, 4))
P[0:3, 0:3] = np.eye(3)

seen_indices = set()



# Iterate over the time steps

# ------------- EKF part 2 --------------------- #


# current_pose = world_T_imu[:, :, 0] # EKF Part 2


# ------------- EKF part 2 --------------------- #







# ------------- EKF part 3 --------------------- #

current_pose = world_T_imu_slam[:, :, 0] # EKF Part 3

# ------------- EKF part 3 --------------------- #







for i in range(t.shape[1] - 1):

	# ------------- EKF Part 3 --------------------- #

	# current_pose = world_T_imu_slam[:, :, i]

	# Get position and angle vector
	pos_vel_vec = np.array([(t[0][i + 1] - t[0][i]) * linear_velocity[:, i][0], (t[0][i + 1] - t[0][i]) * linear_velocity[:, i][1], (t[0][i + 1] - t[0][i]) * linear_velocity[:, i][2], (t[0][i + 1] - t[0][i]) * angular_velocity[:, i][0], (t[0][i + 1] - t[0][i]) * angular_velocity[:, i][1], (t[0][i + 1] - t[0][i]) * angular_velocity[:, i][2]])
	world_T_imu_slam[:, :, i + 1] = current_pose @ twist2pose(axangle2twist(pos_vel_vec.reshape(1, 6))) # Get updated pose

	#next_pose = world_T_imu_slam[:, :, i + 1]


	# Update Covariance
	covar_slam[3 * features.shape[1]:, 3 * features.shape[1]:] = twist2pose(-1 * axangle2adtwist(pos_vel_vec)) @ covar_slam[3*features.shape[1]:, 3*features.shape[1]:] @ np.transpose(twist2pose(-1 * axangle2adtwist(pos_vel_vec))) + W

	# Compute F
	F = twist2pose(-1 * axangle2adtwist(pos_vel_vec))
	# Cov Left - Right
	covar_slam[:3 * features.shape[1], 3 * features.shape[1]:] = covar_slam[:3 * features.shape[1], 3 * features.shape[1]:] @ F.T
	# Cov Right - Left
	covar_slam[3 * features.shape[1]:, :3 * features.shape[1]] = F @ covar_slam[3 * features.shape[1]:, :3 * features.shape[1]]

	# ------------- EKF Part 3 --------------------- #







	# Get good features with no un observed landmark points
	good_features_inx = sorted(list(set(range(features.shape[1])) - set(np.unique(np.where((features[:, :, i].T == [-1, -1, -1, -1]))[0]))))
	good_features = features[:, good_features_inx, i]

	indices_to_pop = set()
	# Analysis at time t for the observations
	for j, idx in enumerate(good_features_inx): # Iterate over the indices of the good features
		if idx not in seen_indices: # If the index is not in the seen indices add to the list
			pixel_coords = features[:, idx, i] #Get the pixel coordinate

			# Get xyz world coordinates from motion model
			z = (-1 * K_s[2, 3]) / (pixel_coords[0] - pixel_coords[2])
			x = (pixel_coords[0] - K_s[0, 2]) * z / K_s[0, 0]
			y = (pixel_coords[1] - K_s[1, 2]) * z / K_s[1, 1]
			optical_landmark_coords_hom = np.array([x, y, z, 1]) # Landmark in homogeneous coordinates

			# Transform from CAM to IMU frame, then from IMU frame to World frame
			landmark_coords_world = current_pose @ imu_T_cam @ optical_landmark_coords_hom

			# Distance between robot and landmark position
			dist_robot_landmark = np.linalg.norm(current_pose[0:2, 3] - landmark_coords_world[0:2])

			indices_to_pop.add(idx) # Add index to list

			if rem_outliers:
				if dist_robot_landmark > outlier_t:
					continue

			# Landmark position initialization
			mu_start[:, idx] = landmark_coords_world

			seen_indices.add(idx)
		else:
			pass # Don't do anything


	# ------------- EKF Part 2 --------------------- #


	# Compute the projections and the Jacobian of the Projections
	# good_obs_indices = list(set(good_features_inx) - indices_to_pop)
	# N_t = len(good_obs_indices)
	# V = np.diag(np.ones((N_t,)) * pert_v)
	# # Only consider the landmarks that were observed at this time step
	# if mu_start[:, good_obs_indices][0].shape[0] == 0:
	# 	continue
	# # Compute the projection pi features.shape[1] by 4
	# pi = projection((cam_T_imu @ np.linalg.inv(world_T_imu[:, :, i + 1]) @ mu_start[:, good_obs_indices]).T)
	#
	# # Get the innovation
	# innovation = features[:, good_obs_indices, i] - (K_s @ np.transpose(pi))
	# # Get the Jacobian of pi for all observations
	# H = projectionJacobian(pi) # Size N_t by 4 by 4
	#
	# H_new = np.zeros((4 * N_t, 3 * features.shape[1]))
	# for j in range(len(good_obs_indices)):
	# 	H_new[4*j:4*j + 4, good_obs_indices[j]:good_obs_indices[j] + 3] = K_s @ H[j, :, :] @ cam_T_imu @ np.linalg.inv(world_T_imu[:, :, i]) @ P.T
	#
	# #print("Created H_new")
	# Kalman_gain = cov_start @ H_new.T @ np.linalg.inv((H_new @ cov_start @ H_new.T) + np.kron(np.eye(4), V))
	#
	# mu_start[0:3] += np.reshape((Kalman_gain @ innovation.flatten(order='F')), (3,features.shape[1]),order='F')
	# cov_start = (np.eye(3*features.shape[1],3*features.shape[1]) - (Kalman_gain @ H_new)) @ cov_start
	# print("End of iteration : ", i)
	#
	# current_pose = world_T_imu[:, :, i + 1]


	# ------------- EKF Part 2 --------------------- #



	# ------------- EKF Part 3 --------------------- #


	good_obs_indices = list(set(good_features_inx) - indices_to_pop)
	N_t = len(good_obs_indices)
	V = np.diag(np.ones((N_t,)) * pert_v)
	H = np.zeros((4 * N_t, 3 * features.shape[1]))
	# Only consider the landmarks that were observed at this time step
	if mu_start[:, good_obs_indices][0].shape[0] == 0:
		continue


	# Compute the projection pi features.shape[1] by 4
	pi = projection((cam_T_imu @ np.linalg.inv(world_T_imu_slam[:, :, i + 1]) @ mu_start[:, good_obs_indices]).T)

	# Get the innovation
	innovation = features[:, good_obs_indices, i] - (K_s @ np.transpose(pi))


	H_slam = np.zeros((4 * N_t, 6 + 3 * features.shape[1]))
	H_robot = np.zeros((4 * N_t, 6))

	# Compute Jacobian
	J = projectionJacobian(pi)

	for j in range(len(good_obs_indices)):
		H[4 * j:4 * j + 4, good_obs_indices[j] * 3:good_obs_indices[j] * 3 + 3] = K_s @ J[j, :, :] @ cam_T_imu @ np.linalg.inv(world_T_imu_slam[:, :, i + 1]) @ P.T
		H_robot[j * 4: j * 4 + 4, :] = -1 * K_s @ J[j, :, :] @ cam_T_imu  @ op(np.linalg.inv(world_T_imu_slam[:, :, i + 1]) @ mu_start[:, good_obs_indices[j]])

	# Concatenate
	H_slam[:, 3 * features.shape[1]:] = H_robot
	H_slam[:, :3 * features.shape[1]] = H

	# Compute Kalman Gain
	Kalman_gain_slam = covar_slam @ H_slam.T @ np.linalg.inv((H_slam @ covar_slam @ H_slam.T) + np.kron(np.eye(4), V))

	# Update
	mu_start[0:3] += np.reshape((Kalman_gain_slam[:3 * features.shape[1], :] @ innovation.flatten(order='F')),(3, features.shape[1]), order='F')

	pose_updated_splitted = np.reshape(Kalman_gain_slam[3 * features.shape[1]:, :] @ innovation.flatten(order='F'), (2, 3))
	first = pose_updated_splitted[0]
	second = pose_updated_splitted[1]
	arg = np.concatenate((first, second), axis=0)

	# Get pose for measurements up to t + 1
	world_T_imu_slam[:, :, i + 1] = world_T_imu_slam[:, :, i + 1] @ twist2pose(axangle2twist(arg.reshape(1, 6)))

	# Covariance update up to t + 1
	covar_slam = (np.eye(3 * features.shape[1] + 6, 3 * features.shape[1] + 6) - (Kalman_gain_slam @ H_slam)) @ covar_slam
	covar_slam = (covar_slam + covar_slam.T) / 2 # Ensures that the matrix remains symmetric

	# Update for next iteration a pose computed up to measurements t + 1
	current_pose = world_T_imu_slam[:, :, i + 1]
	print("End of iteration : ", i)

	# ------------- EKF Part 3 --------------------- #


# ------------- EKF Part 2 --------------------- #

# fig, ax = plt.subplots()
# ax.plot(world_T_imu[0,3, :], world_T_imu[1,3, :], color='red')
# #visualize_trajectory_2d(world_T_imu, show_ori = True)
# ax.scatter(mu_start[0], mu_start[1], s=3)
# plt.show()

# ------------- EKF Part 2 --------------------- #



# ------------- EKF Part 3 --------------------- #

fig, ax = plt.subplots()
ax.plot(world_T_imu_slam[0,3, :], world_T_imu_slam[1,3, :], color='red')
#visualize_trajectory_2d(world_T_imu, show_ori = True)
ax.scatter(mu_start[0], mu_start[1], s=3)
plt.show()

# ------------- EKF Part 3 --------------------- #





