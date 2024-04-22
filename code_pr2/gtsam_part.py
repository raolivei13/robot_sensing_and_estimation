import gtsam
import plotly.graph_objects as go
from load_data import get_data
from scipy.special import expit
import cv2

from pr2_utils import bresenham2D
import matplotlib
import transforms3d as t3d
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
from icp_warm_up import icp_file
#from part_one import x_next # This is getting Poses from dead reckoning

# Set Data Set
dataset = 21

# Image paths for loading RGBD and Disparity Images
disp_path = "dataRGBD/Disparity" + str(dataset) + "/"
rgb_path = "dataRGBD/RGB" + str(dataset) + "/"


x_opt_next = np.load('opt_icp_traj.npy') # Load ICP trajectory


def load_disp_rgb(disp_path, rgb_path, dataset, im_num):

    imd = cv2.imread(disp_path + 'disparity' + str(dataset) + '_' + str(im_num) + '.png', cv2.IMREAD_UNCHANGED)  # (480 x 640) # Disparity image
    imc = cv2.imread(rgb_path + 'rgb' + str(dataset) + '_' + str(im_num) + '.png')[..., ::-1]  # (480 x 640 x 3) # RGB image

    # print(imc.shape)

    # convert from disparity from uint16 to double
    disparity = imd.astype(np.float32)

    #print(disparity.shape)

    # get depth
    dd = (-0.00304 * disparity + 3.31)
    z = 1.03 / dd

    # calculate u and v coordinates
    v, u = np.mgrid[0:disparity.shape[0], 0:disparity.shape[1]]
    # u,v = np.meshgrid(np.arange(disparity.shape[1]),np.arange(disparity.shape[0]))

    # get 3D coordinates
    fx = 585.05108211
    fy = 585.05108211
    cx = 315.83800193
    cy = 242.94140713
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z

    # calculate the location of each pixel in the RGB image
    rgbu = np.round((u * 526.37 + dd * (-4.5 * 1750.46) + 19276.0) / fx)
    rgbv = np.round((v * 526.37 + 16662.0) / fy)
    valid = (rgbu >= 0) & (rgbu < disparity.shape[1]) & (rgbv >= 0) & (rgbv < disparity.shape[0])

    #imc[rgbv[valid].astype(int), rgbu[valid].astype(int)] / 255.0

    # Returns stacked rgbu and rgbv pixels - valid - original RGB image
    return np.stack((rgbu, rgbv, np.ones((rgbu.shape)) * z), axis=2), valid, imc


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


# -------------------------------------- GTSAM part --------------------------------------- #


# Create Factor graph


graph = gtsam.NonlinearFactorGraph()
prior_model = gtsam.noiseModel.Diagonal.Sigmas((0.3, 0.3, 0.3))
graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0, 0, 0), prior_model))



odometry_model = gtsam.noiseModel.Diagonal.Sigmas((0.2, 0.2, 0.1))
Between = gtsam.BetweenFactorPose2
initial_estimate = gtsam.Values()

for i in range(x_opt_next.shape[1] - 2):
    rel_pose = np.linalg.inv(create_pose(create_rotation_along_z(x_opt_next[:, i][2]), np.array([x_opt_next[:, i][0], x_opt_next[:, i][1], 0]))) @ create_pose(create_rotation_along_z(x_opt_next[:, i + 1][2]), np.array([x_opt_next[:, i + 1][0], x_opt_next[:, i + 1][1], 0]))
    graph.add(Between(i + 1, i + 2, gtsam.Pose2(rel_pose[0, 3], rel_pose[1, 3], t3d.euler.mat2euler(rel_pose[0:3, 0:3])[2]), odometry_model))


# make connections every 10 nodes
skip = 10
# Establish fixed inetrval loop closure every 10 poses

# p = 1

sub = x_opt_next.shape[1] % 10


for i in range(0, x_opt_next.shape[1] - sub, skip):

    rel_pose = np.linalg.inv(
            create_pose(create_rotation_along_z(x_opt_next[:, i][2]), np.array([x_opt_next[:, i][0], x_opt_next[:, i][1], 0]))) @ create_pose(
            create_rotation_along_z(x_opt_next[:, i + 9][2]), np.array([x_opt_next[:, i + 9][0], x_opt_next[:, i + 9][1], 0]))
    graph.add(Between(i + 1, i + 10, gtsam.Pose2(rel_pose[0, 3], rel_pose[1, 3], t3d.euler.mat2euler(rel_pose[0:3, 0:3])[2]), odometry_model))


    for k in range(skip - 2):
        rel_pose = np.linalg.inv(create_pose(create_rotation_along_z(x_opt_next[:, k + 1][2]), np.array([x_opt_next[:, k + 1][0], x_opt_next[:, k + 1][1], 0]))) @ create_pose(
                    create_rotation_along_z(x_opt_next[:, (i + skip) - 1][2]), np.array([x_opt_next[:, (i + skip) - 1][0], x_opt_next[:, (i + skip) - 1][1], 0]))
        graph.add(Between(k + 2, i + skip, gtsam.Pose2(rel_pose[0, 3], rel_pose[1, 3], t3d.euler.mat2euler(rel_pose[0:3, 0:3])[2]),odometry_model))


    # p += 1

# Create initial estimates
for i in range(x_opt_next.shape[1] - 1):
    initial_estimate.insert(i + 1, gtsam.Pose2(x_opt_next[:, i][0], x_opt_next[:, i][1], x_opt_next[:, i][2]))



# Optimize with Gauss Newton
optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate)
result = optimizer.optimize()



fig = go.Figure()
fig.add_scatter(x = x_opt_next[0, :], y = x_opt_next[1, :], name="initial", marker=dict(color='orange'))
final_poses = gtsam.utilities.extractPose2(result)
fig.add_scatter(x = final_poses[:, 0], y = final_poses[:, 1], name="optimized", marker=dict(color='green'))
fig.update_yaxes(scaleanchor = "x",scaleratio = 1)
fig.show()

# -------------------------------------- GTSAM part --------------------------------------- #

#final_poses = np.transpose(final_poses)


# -------------------------------------- Load Data & sync data --------------------------------------- #


#Gets encoder, IMU and Lidar data
data_encoder, data_hok, data_imu, data_kin = get_data(dataset)


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

# Get Kinect Data Time stamps
disp_stamps = data_kin[0]  # acquisition times of the disparity images
rgb_stamps = data_kin[1]  # acquisition times of the rgb images

lidar_stamps = data_hok[6]  # acquisition times of the lidar scans


#Sync data for imu
new_imu_angular_velocity = np.zeros((3, len(encoder_stamps)))
new_imu_stamps = np.zeros(len(encoder_stamps))

#Sync data for LiDar
new_lidar_ranges = np.zeros((lidar_ranges.shape[0], len(encoder_stamps)))
new_lidar_stamps = np.zeros(len(encoder_stamps))

print("Data imported")

print("Starting Sync of encoder with IMU and LiDar")
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


print("End sync")


# print("Starting Sync of RGB")


# -------------------------------------- Load Data & sync data --------------------------------------- #



# list_encoder_rgb_ts = []
# # get closest timestamp from imu data to align to encoder ts
# for i in range(len(rgb_stamps)):
#     ts_enc_diff_kinect = encoder_stamps - rgb_stamps[i]
#     arg = np.argmin(np.abs(ts_enc_diff_kinect))
#     list_encoder_rgb_ts.append(arg)
#
# # final_poses = np.transpose(final_poses)
# poses_reduced = final_poses[:, list_encoder_rgb_ts] # Get pose array
#
#
# poses_reduced = np.zeros((3, final_poses.shape[1]))
# for i in range(len(list_encoder_rgb_ts)):
#
#     poses_reduced

# print("End sync")






# ----------------- Create occupancy Grid with the entire LiDar Scans --------------------------------- #

print("Start Occupancy grid")

res = 0.05 # Resolution

if dataset == 20:
    n1 = 1000
    n2 = 1000
else:
    n1 = 2000
    n2 = 2000



grid = np.ones((n1, n2))
#res = 0.05 # Resolution
grid_center_x = int(grid.shape[0]/2 - 1)
grid_center_y = int(grid.shape[1]/2 - 1)


#Create an occupancy grid with the Odometry data
#Iterate over the entire the time stamps

final_poses = final_poses.T

for i in range(len(encoder_stamps) - 1):

    lidar_measure = np.delete(new_lidar_ranges[:, i], [np.argmin(new_lidar_ranges[:, i]), np.argmax(new_lidar_ranges[:, i])])  # Dim 1081 x 1
    lidar_measure_rect = convert_to_rect(lidar_measure, lidar_angle_increment[0][0], lidar_angle_min)  # This is N by 3

    # Starting point of LiDar in pixel frame
    lidar_vec_start = np.array([0, 149.165 / 1000, 0]) / res

    # Lidar end points convert to World Frame
    R = np.array([[np.cos(final_poses[:, i][2]), -np.sin(final_poses[:, i][2]), 0],
                  [np.sin(final_poses[:, i][2]), np.cos(final_poses[:, i][2]), 0],
                  [0, 0, 1]])
    p = np.array([final_poses[:, i][0], final_poses[:, i][1], 0])
    lidar_measure_rect_W = np.matmul(R, lidar_measure_rect.T) + p.reshape((3, 1))

    for j in range(lidar_measure_rect.shape[0]):
        filled_pixels = bresenham2D(lidar_vec_start[0], lidar_vec_start[1], lidar_measure_rect_W[:, j][0] / res,
                                    lidar_measure_rect_W[:, j][1] / res)
        grid[grid_center_x + filled_pixels[0].astype(int), grid_center_y + filled_pixels[1].astype(int)] = 0
        # Decrease / Increase Log odds
        grid[grid_center_x + filled_pixels[0, :-1].astype(int), grid_center_y + filled_pixels[1, :-1].astype(
            int)] -= np.log(4)
        grid[grid_center_x + filled_pixels[0, -1].astype(int), grid_center_y + filled_pixels[1, -1].astype(
            int)] += np.log(4)

    print("Iteration :", i)

print("End Occupancy grid")

occ_grid_sigmoid = expit(grid)
plt.imsave(fname='d'+str(dataset)+'_occ_grid_full_GTSAM.png',arr=np.flip(occ_grid_sigmoid,axis=1).T,dpi=600)


# --------------------------------------Texture Map --------------------------------------- #


list_encoder_rgb_ts = []
# get closest timestamp from imu data to align to encoder ts
for ts in rgb_stamps:
    ts_enc_diff_kinect = encoder_stamps - ts
    arg = np.argmin(np.abs(ts_enc_diff_kinect))
    list_encoder_rgb_ts.append(arg)

# final_poses = np.transpose(final_poses)
# poses_reduced = final_poses[list_encoder_rgb_ts] # Get pose array

poses_reduced = np.zeros((final_poses.shape[1], 3))
for i in range(len(list_encoder_rgb_ts)):

    if list_encoder_rgb_ts[i] == 4955:
        continue

    poses_reduced[i, :] = final_poses[:, i]



yaw = 0.021
picth = 0.36

# Create rotation matrices
Rz = create_rotation_along_z(yaw)
Ry = create_rotation_along_y(picth)
# Position of camera in body frame
p_cam_in_body = np.array([(115.1 + 2.66)/1000, 0 , 380.1/1000])

# Body to cam pose
body_T_cam = create_pose(Ry @ Rz, p_cam_in_body)

# Optical to rgeular camera frame and inverse
optical_R_reg = np.array([[0, -1, 0],[0, 0, -1], [1, 0, 0]])
optical_R_reg_inv = np.linalg.inv(optical_R_reg)

# intrinsic parameters of the depth camera and invers
K = np.array([[585.05108211, 0, 242.94140713],
              [0, 585.05108211, 315.83800193],
              [0, 0, 1]])

K_inv = np.linalg.inv(K)


z_threshold = 0.2 # Set z - axis theshold

occ_grid_rgb = np.zeros((1130, 1130, 3))
grid_centre_x = int(1130 / 2 - 1)
grid_centre_y = int(1130 / 2 - 1)

final_poses = final_poses.T

print("Start Creating texture map")
for i in range(rgb_stamps.shape[0] - 1):

    # Load the image
    rgbu_rgbv_stacked, valid, imc = load_disp_rgb(disp_path, rgb_path, dataset, i + 1)

    # Compute pose transformation from robot frame to world frame
    robot_pose = create_pose(create_rotation_along_z(poses_reduced[i, :][2]), np.array([poses_reduced[i, :][0], poses_reduced[i, :][1], 0]))

    optical_frame = np.einsum('ji,mni -> mnj', K_inv, rgbu_rgbv_stacked)
    # convert optical to regular camera frame
    regular_frame = np.einsum('ji,mni -> mnj', optical_R_reg_inv, optical_frame)
    # convert camera frame to body
    regular_hom_frame = np.vstack((np.ones((1, 640, 480)), regular_frame.T)).T[:, :, [1,2,3,0]]
    body_frame = np.einsum('ji,mni -> mnj', body_T_cam, regular_hom_frame)
    # convert body frame to world
    world_frame = np.einsum('ji,mni -> mnj', robot_pose, body_frame)
    x_inds, y_inds = np.where((world_frame[:, :, 2] < z_threshold) & (world_frame[:, :, 0] >= 0) & (world_frame[:, :, 1] >= 0))[0], \
                     np.where((world_frame[:, :, 2] < z_threshold) & (world_frame[:, :, 1] >= 0) & (world_frame[:, :, 0] >= 0))[1]
    valid_xyz = world_frame[x_inds, y_inds]
    valid_xyz = (valid_xyz/res).astype(int)
    valid_xyz[:, 0] += grid_centre_x
    valid_xyz[:, 1] += grid_centre_y

    occ_grid_rgb[valid_xyz[:, 0], valid_xyz[:, 1]] = imc[x_inds, y_inds]

print("End texture map")
plt.imsave(fname='d'+str(dataset)+'_full_texture_map_GTSAM_time_{x}.png'.format(x=i), arr=np.flip(occ_grid_rgb.astype('uint8'),axis=1), dpi=600)

# --------------------------------------Texture Map --------------------------------------- #
