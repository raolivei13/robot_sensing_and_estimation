import numpy as np
from load_data import get_data
#from scan_matching import create_pose, create_rotation_along_z, create_rotation_along_yk
import matplotlib.pyplot as plt
import cv2

# Load optimized ICP trajectory
x_opt_next = np.load('opt_icp_traj.npy')



# Modify this for data set number
dataset = 20
res = 0.06

# Image paths for loading RGBD and Disparity Images
disp_path = "dataRGBD/Disparity"+str(dataset)+"/"
rgb_path = "dataRGBD/RGB20"+str(dataset)+"/"


def normalize(img):
    max_ = img.max()
    min_ = img.min()
    return (img - min_) / (max_ - min_)




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


# Load data function
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



# -------------------- Load encoeder data , IMU data and Kinect data time stamps -------------------- #

#Gets encoder, IMU and Lidar data
data_encoder, _, data_imu, data_kin = get_data(dataset)

# Get encoder data
encoder_counts = data_encoder[0]
encoder_stamps = data_encoder[1]

# Get IMU data
imu_angular_velocity = data_imu[0]
imu_stamps = data_imu[2]

# Get Kinect Data Time stamps
disp_stamps = data_kin[0]  # acquisition times of the disparity images
rgb_stamps = data_kin[1]  # acquisition times of the rgb images

# -------------------- Load encoeder data , IMU data and Kinect data -------------------- #








# Get Poses from optimized trajectory by ICP to convert data from Body - Robot frame to World Frame coordinate
# For now just use Dead - Reckoning trajectory

list_encoder_rgb_ts = []
# get closest timestamp from imu data to align to encoder ts
for ts in rgb_stamps:
    ts_enc_diff_kinect = encoder_stamps - ts
    arg = np.argmin(np.abs(ts_enc_diff_kinect))
    list_encoder_rgb_ts.append(arg)

x_opt_next = np.transpose(x_opt_next)
poses_reduced = x_opt_next[list_encoder_rgb_ts] # Get pose array

# Start loading data


# Iterate over the trajectory length
# Load each image one at a time
# Convert to Optical Frame
# Convert to Regular camera frame coordinates
# Convert to Body - Robot frame coordinates
# Convert from Body - Robot frame coordinates to World frame coordinates using robot pose


# construct disparity camera to world transformation
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
plt.imsave(fname='d20_full_texture_map_time_{x}.png'.format(x=i), arr=np.flip(occ_grid_rgb.astype('uint8'),axis=1), dpi=600)


