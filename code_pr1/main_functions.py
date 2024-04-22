import pickle
import sys
import time

from numpy import array, zeros, exp, cos, arccos, sin, dot, cross, log, sum, pi, append
import autograd.numpy as np
from autograd import grad
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Choose the appropriate backend, e.g., TkAgg or Qt5Agg




#Calibrate IMU measurements
def calibrate_IMU(imud, V_ref, sens_gyro_deg, sens_acc):


    #Iterate through the 9 IMU datasets
    new_imud = [] #List of calibrated IMU measurements
    for i in range(len(imud)):


        #Access the individual Dat sets
        curr_imu_set = imud[i]


        curr_imu_set_vals = curr_imu_set["vals"] #Look at values

        #Invert Signs of x and y Acceleration respectively
        for j in range(curr_imu_set_vals.shape[1]):


            # Invert signs
            curr_imu_set_vals[:, j][0] = -1 * curr_imu_set_vals[:, j][0]
            curr_imu_set_vals[:, j][1] = -1 * curr_imu_set_vals[:, j][1]
            #curr_col_imu[0] = -1 * curr_col_imu[0] #Invert sign of first comp
            #curr_col_imu[1] = -1 * curr_col_imu[1] #Invert sign of 2nd comp

            #Copy Back to original matrix
            #curr_imu_set_vals[:, j] = curr_col_imu

        # num_bias_elems = 600
        bias_curr = (np.sum(curr_imu_set_vals, axis=1) / curr_imu_set_vals.shape[1]).round(1) #Compute Bias
        scale_factor_acc = V_ref / (1023 * sens_acc)
        scale_factor_gyro = V_ref / (1023 * sens_gyro_deg) * (np.pi / 180)

        # Compute new IMU calibrated values
        new_imud_set = np.zeros((curr_imu_set_vals.shape[0], curr_imu_set_vals.shape[1]))
        for k in range(curr_imu_set_vals.shape[1]):
        #for k in range(num_bias_elems):
            col_empty = new_imud_set[:, k] #Get the empty new_imud_set column to populate
            col_curr_imu = curr_imu_set_vals[:, k]


            #Final processed value
            # curr_imu_set_vals[:, k][:2] = (col_curr_imu[:2] - bias_curr[:2]) * scale_factor_acc
            # curr_imu_set_vals[:, k][2] = (col_curr_imu[2] - (bias_curr[2] + (1 / scale_factor_acc))) * scale_factor_acc
            # curr_imu_set_vals[:, k][-3:] = (col_curr_imu[-3:] - bias_curr[-3:]) * scale_factor_gyro

            col_empty[:2] = (col_curr_imu[:2] - bias_curr[:2]) * scale_factor_acc
            col_empty[2] = (col_curr_imu[2] - (bias_curr[2] + (1 / scale_factor_acc))) * scale_factor_acc
            col_empty[-3:] = (col_curr_imu[-3:] - bias_curr[-3:]) * scale_factor_gyro

            # #Copy back
            new_imud_set[:, k] = col_empty

        #new_imud_set[:, num_bias_elems:] = curr_imu_set_vals[:, num_bias_elems:]
        #Store calibarated IMU measurements in the new set
        #new_imud.append(curr_imu_set_vals)
        new_imud.append(new_imud_set)

    return new_imud





#Motion Model predicting quaternion at next time step
def motion_model(ang_vel, quat_start, time_diff):

    #Angular velocity is assumed to organized

    # Small Perturbation added to avoid singularities
    ang_vel = ang_vel + 1e-3

    exp_comp = exp_map(np.append(0.0, (time_diff / 2)*ang_vel))

    return quat_mul(quat_start, exp_comp)


#Observation model
def obs_model(quat_start):

    # quat_start = quat_start + 1e-3

    #g = np.array([0, 0, 0, -9.81])
    g = np.array([0, 0, 0, 1])

    first_mul = quat_mul(quat_inv(quat_start), g)
    acc_t = quat_mul(first_mul, quat_start)

    return acc_t[-3:]


#Recovers the rotation angle theta from the quaternion
def log_quat(quat):

    quat = quat + 1e-3 #In order to avoid singularities

    if (quat[1:] < 0.0005).all():
        theta_vec = 2 * np.array([np.log(np.abs(quat[0])), 0.0, 0.0, 0.0])
    else:
        #norm_vec_tot = norm(quat) #Total Quaternion 2 - norm
        norm_vec_tot = np.sqrt(np.dot(quat, quat))
        #norm_vec_part = norm(quat[-3:]) #Vector portion of Quaternion 2 - norm
        norm_vec_part = np.sqrt(np.dot(quat[-3:], quat[-3:]))

        first_comp = np.log(norm_vec_tot)
        second_comp = (quat[-3:] / norm_vec_part) * np.arccos(quat[0] / norm_vec_tot)

        #Log Quat vector
        log_q = np.append(first_comp, second_comp)

        #Actual theta vector
        theta_vec = 2*log_q



    return theta_vec[-3:]

# Just performs quaternion multiplication
def quat_mul(q, p):

    first_comp = q[0]*p[0] - np.dot(q[-3:], p[-3:])
    second_comp = q[0]*p[-3:] + p[0]*q[-3:] + np.cross(q[-3:], p[-3:])

    #Result
    return np.append(first_comp, second_comp)

def quat_inv(q):

    q_conj = np.append(q[0], -1*q[-3:])

    return q_conj / (np.dot(q, q))

def re_order(vec):
    #Re - Order gyroscope measurements
    return np.array([vec[1], vec[2], vec[0]])

def exp_map(quat):

    #quat = quat + 0.001

    first_comp = np.cos(np.sqrt(np.dot(quat[-3:], quat[-3:])))
    second_comp = (quat[-3:] / np.sqrt(np.dot(quat[-3:], quat[-3:])))*np.sin(np.sqrt(np.dot(quat[-3:], quat[-3:])))

    return np.exp(quat[0])*np.append(first_comp, second_comp)




















