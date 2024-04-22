from main_functions import calibrate_IMU, motion_model, obs_model, log_quat, quat_mul, quat_inv, re_order
from load_data import upload_data, tic, toc

from numpy import array, zeros, exp, cos, arccos, sin, dot, cross, log, sum, pi, append, transpose, newaxis
import autograd.numpy as np
from autograd import grad
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Choose the appropriate backend, e.g., TkAgg or Qt5Agg

#Specs
im_height = 240
im_width = 320
w_angle = 60
h_angle = 45
del_alpha = w_angle / im_width
del_beta = h_angle / im_height
im_step_size = 10
pan_height = 2100
pan_width = 3000

#We need to divide the images into 4 quadrants
#Look at different case when:
# pixel is in 1st, 2nd, 3rd, 4th quadrant

#Reverse Data function for VICON data and Pose Matrices
def reverse_data(data, num_data):
    new_data = np.zeros((num_data, 3, 3))
    for i in range(num_data):
        new_data[i] = data[:, :, i]

    return new_data


def long_lat_coor():

    lon_lat_map = np.zeros((im_height, im_width, 2))
    for i in range(im_height):
        for j in range(im_width):

            #1st quadrant
            if i <= (im_height - 1) / 2 and j > (im_width - 1) / 2:
                long = -del_alpha * (j - (im_width / 2))
                lat = -del_beta * (i - (im_height / 2))

            #2nd quadrant
            elif i <= (im_height - 1) / 2 and j < (im_width - 1) / 2:
                long = del_alpha * (j - (im_width / 2))
                lat = -del_beta * (i - (im_height / 2))

            #3rd quadrant
            elif i > (im_height - 1) / 2 and j <= (im_width - 1) / 2:
                long = del_alpha * (j - (im_width / 2))
                lat = del_beta * (i - (im_height / 2))

            #4th quadrant
            elif i > (im_height - 1) / 2 and j > (im_width - 1) / 2:
                long = -del_alpha * (j - (im_width / 2))
                lat = del_beta * (i - (im_height / 2))

            lon_lat_map[i, j] = [lat, long]

    return lon_lat_map

#Rotate Cartesian coordinates to World frame
#Specify Cartesian coordinate block and Pose matrix T
def rot_cart_to_w(cart_block, T):

    rot_cart_im = np.zeros((im_height, im_width, 4, int(np.floor(cam_ims.shape[-1] / im_step_size))+1))
    camd_red = np.zeros((im_height, im_width, 3, int(np.floor(cam_ims.shape[-1] / im_step_size))+1))

    idx = 0
    for i in range(0, len(T), im_step_size):

        rot_cart_im[:, :, :, idx] = np.einsum('rr,mnr->mnr', T[i], cart_block)
        camd_red[:, :, :, idx] = cam_ims[:, :, :, i]

        idx += 1

    return rot_cart_im, camd_red


#Rotate Cartesian back to spherical coordinates
def rotate_cart_to_spher(rot_cart_im):

    arr_im_sphere = np.zeros((im_height, im_width, 2, int(np.floor(cam_ims.shape[-1] / im_step_size)) + 1))

    for i in range(rot_cart_im.shape[-1]):
        arr_im_sphere[:, :, :, i][:, :, 1] = np.arctan(rot_cart_im[:, :, :, i][:, :, 0] / rot_cart_im[:, :, :, i][:, :, 2])
        arr_im_sphere[:, :, :, i][:, :, 0] = np.arctan(rot_cart_im[:, :, :, i][:, :, 0] / (np.sqrt(rot_cart_im[:, :, :, i][:, :, 0] ** 2 + rot_cart_im[:, :, :, i][:, :, 2] ** 2)))

    return arr_im_sphere


#Cylinder method
def cyliner_method(arr_im_sphere):

    long_pixs_per_rad = pan_width / (
                arr_im_sphere[:, :, :, :][:, :, 1].max() - arr_im_sphere[:, :, :, :][:, :, 1].min() + 0.09)
    lat_pixs_per_rad = pan_height / (
                arr_im_sphere[:, :, :, :][:, :, 0].max() - arr_im_sphere[:, :, :, :][:, :, 0].min() + 0.09)

    center = arr_im_sphere[:, :, :, 0][im_height // 2, im_width // 2]

    long_center_per_rad = (arr_im_sphere[:, :, :, :][:, :, 1].max() + arr_im_sphere[:, :, :, :][:, :, 1].min()) / 2
    lat_center_per_rad = (arr_im_sphere[:, :, :, :][:, :, 0].max() + arr_im_sphere[:, :, :, :][:, :, 0].min()) / 2

    return long_pixs_per_rad, lat_pixs_per_rad, center, long_center_per_rad, lat_center_per_rad





#Load Data
camd, imud, vicd = upload_data() #Need Cam Data



cam_data_nums = [1, 2, 8, 9, 10, 11] #Cam data indices

#Start iterating through the Cam Data Sets
for i in range(len(cam_data_nums)):


    if cam_data_nums[i] == 1 or cam_data_nums[i] == 2 or cam_data_nums[i] == 8 or cam_data_nums[i] == 9:
        # -------------------------------------- Index i is being used ------------------------------
        #Extract contents from VICON Data
        vic_rotations = vicd[cam_data_nums[i] - 1]['rots'] #Get Rotations
        vic_time = vicd[cam_data_nums[i] - 1]['ts'][0] #Get time stamps


        # Extract contents from CAM Data
        cam_ims = camd[i]['cam'] #Get images
        cam_time = camd[i]['ts'][0] #Get time Stamps

        num_vicd = vic_rotations.shape[2]
        # -------------------------------------- Index i is being used ------------------------------

    else:
        #Extract contents from VICON Data
        vic_rotations = vicd[8]['rots'] #Get Rotations
        vic_time = vicd[8]['ts'][0]

        # Extract contents from CAM Data
        cam_ims = camd[i]['cam'] #Get images
        cam_time = camd[i]['ts'][0]

        num_vicd = vic_rotations.shape[2]



    #Get Long Lat Transformations and convert to radians
    lon_lat_map = long_lat_coor() * (np.pi / 180)

    #Reverse VICON Data
    vicon_red = reverse_data(vic_rotations, num_vicd)

    #Align Cam Time with Vicon Time
    list_args = []
    for j in range(len(cam_time)):
        diff = vic_time - cam_time[j]
        arg_nearest = len(diff[diff <= 0]) - 1
        if arg_nearest < num_vicd:
            list_args.append(arg_nearest)


    #vic_rotations_new = vic_rotations[:, :, list_args] #Get new rotations that are time alligne
    vic_rotations_new = vicon_red[list_args]

    #We define the pose Transformation
    T = np.zeros((vic_rotations_new.shape[0], 4, 4))
    for p in range(vic_rotations_new.shape[0]):
        for m in range(4):
            for n in range(4):
                T[p, :, :][0:3, 0:3] = vic_rotations_new[p, :, :] #Insert Rotation Matrix
                T[p, :, :][3, 3] = 1 #By definition
                T[p, :, :][2, 3] = 0.1 #Relative position of the camera wrt to the IMU is 0.1 m above the IMU


    #We convert the from spherical to homogeneous Cartesian coordinates
    cart_block = np.zeros((im_height, im_width, 4)) #Depth 4, array of 4x1 dimension fits through
    for m in range(im_height):
        for n in range(im_width):

            #Regular Spherical to Cartesian coordinates transformation
            x = np.sin(lon_lat_map[m, n][0]) * np.cos(lon_lat_map[m, n][1])
            y = np.sin(lon_lat_map[m, n][0]) * np.sin(lon_lat_map[m, n][1])
            z = np.cos(lon_lat_map[m, n][0])

            cart_block[m, n] = np.array([x, y, z, 1])



    rot_cart_im, camd_red = rot_cart_to_w(cart_block, T) #Rotate Cartesian to World Coordinate frame

    arr_im_sphere = rotate_cart_to_spher(rot_cart_im) #Rotate from cartesian to spherical



    #Define empty Panorama

    pan = np.zeros((pan_height, pan_width, 3))


    #Cylinder method
    long_pixs_per_rad, lat_pixs_per_rad, center, long_center_per_rad, lat_center_per_rad = cyliner_method(arr_im_sphere)

    pan_cntr_long = (pan_width / 2) - 1
    pan_cntr_lat = (pan_height / 2) - 1

    print("Start Panorama")
    for j in range(arr_im_sphere.shape[-1]):

        lat_long_ref = arr_im_sphere[:, :, :, j]
        #Actual RGB Image
        actual_img = camd_red[:, :, :, j]

        for m in range(lat_long_ref.shape[0]):
            for n in range(lat_long_ref.shape[1]):

                pix_ang_cent2lat = lat_long_ref[m, n][0] - lat_center_per_rad
                pix_ang_cent2long = lat_long_ref[m, n][1] - long_center_per_rad

                pan_pix_4centrvert = pix_ang_cent2lat * lat_center_per_rad
                pan_pix_4centrhor = pix_ang_cent2long * long_pixs_per_rad

                if pan_pix_4centrvert > 0 and pan_pix_4centrhor > 0:
                    pan_lat = pan_cntr_lat - round(pan_pix_4centrvert)
                    pan_long = pan_cntr_long - round(pan_pix_4centrhor)

                elif pan_pix_4centrvert > 0 and pan_pix_4centrhor < 0:
                    pan_lat = pan_cntr_lat - round(pan_pix_4centrvert)
                    pan_long = pan_cntr_long + round(pan_pix_4centrhor) - 1

                elif pan_pix_4centrvert < 0 and pan_pix_4centrhor > 0:
                    pan_lat = pan_cntr_lat + round(pan_pix_4centrvert) - 1
                    pan_long = pan_cntr_long - round(pan_pix_4centrhor)

                elif pan_pix_4centrvert < 0 and pan_pix_4centrhor < 0:
                    pan_lat = pan_cntr_lat + round(pan_pix_4centrvert) - 1
                    pan_long = pan_cntr_long + round(pan_pix_4centrhor) - 1

                pan_lat = int(pan_lat)
                pan_long = int(pan_long)


                pan[pan_lat, pan_long, :] = actual_img[m, n, :]


    pan = pan.astype(int)

    plt.imshow(pan)
    plt.title(f'Panorama for Cam Data Set number {cam_data_nums[i]}')
    plt.show()



    print("Done Panorama")






