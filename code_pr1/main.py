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


#Cost Function definition
def cost_func(quat):

    #Observation Model portion
    # obs_model_array = np.zeros((quat.shape[0], 4))

    obs_model_array = np.array([obs_model(q) for q in quat])

    # for i in range(quat.shape[0]):
    #     res = obs_model(quat[i, :])
    #     # print(res.shape)
    #     obs_model_array[i, :] = res.reshape(1, -1)

    obs_err = 0.5 * np.sum(np.linalg.norm(np.transpose(new_imud1[:3]) - obs_model_array, axis=1)**2) #Total Observation model error

    #Motion Model portion
    initial_time = 0.0
    time_vector_imu = np.append(initial_time, time_imu_new)

    mot_model_array = np.array(
        [log_quat(quat_mul(quat_inv(quat[i + 1, :]), motion_model(re_order(new_imud1[:, i][-3:]), quat[i, :], time_vector_imu[i + 1] - time_vector_imu[i]))) for i in
         range(quat.shape[0] - 1)])

    #mot_model_array = np.vstack((2 * log_quat(quat_mul(quat_inv(quat[0, :]), motion_model(re_order(new_imud1[:, 0][-3:]), quat[0, :], time_imu_new[1] - 0.0))), mot_model_array))
    # mot_model_array = np.zeros((quat.shape[0], 4))
    # tau = time_imu_new[0] - 0.0
    # for i in range(quat.shape[0] - 1):
    #     ang_vel = new_imud1[:, i][-3:]
    #     ang_vel_new = np.array([ang_vel[1], ang_vel[2], ang_vel[0]])
    #     q_pred = motion_model(ang_vel_new, quat[i, :], tau) #Make prediction
    #
    #     mot_model_array[i, :] = 2 * log_quat(quat_mul(quat_inv(quat[i + 1, :]), q_pred))
    #
    #     #Update
    #     tau = time_imu_new[i + 1] - time_imu_new[i]

    motion_error = 0.5 * np.sum(np.linalg.norm(mot_model_array, axis=1)**2) #Total Motion model error

    return obs_err + motion_error






#Load Data
camd, imud, vicd = upload_data()


#Specifications
V_ref = 3300
sens_gyro_deg = 3.33
sens_acc = 300

#Get IMU Calibrated Data Sets
new_imud = calibrate_IMU(imud, V_ref, sens_gyro_deg, sens_acc)


#Set Number of iterations for PGD
num_iters = 7


#Check Calibration for all 9 Data Sets
#Change the for loop range in this way
# for i in range(len(new_imud) - 2) just see Data Sets 1 to 9
# for i in range(9, len(new_imud)) just see Data Sets 10 and 11
for i in range(len(new_imud) - 2):

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 8))


    #COMMENT HERE
    vicon1 = vicd[i] #Get VICON DataSet
    rots = vicon1['rots']  # Get Rotations
    time = vicon1['ts']  # Get Time
    # Access rots values
    row1 = rots[0]
    row2 = rots[1]
    row3 = rots[2]
    #COMMENT HERE

    new_imud1 = new_imud[i] #Get IMU Data Set
    imud1_time = imud[i]
    vals_imu = new_imud1
    time_imu =imud1_time['ts']
    time_imu_new = time_imu[0]


    #COMMENT HERE
    angles = zeros((3, row1.shape[1] - 1))
    #COMMENT HERE

    angles_imu = zeros((3, len(time_imu_new) - 1))

    #Initialize
    time_curr = 0
    quat_start = array([1, 0, 0, 0])
    tau = time_imu_new[0] - time_curr

    #Initialization for PGD
    #print(new_imud1.shape[1])
    Q = np.zeros((new_imud1.shape[1], 4))
    Q[0, 0] = 1.0 #First vector is initialized to 1, 0, 0, 0

    for j in range(len(time_imu_new) - 1):

        vals_imu_col = vals_imu[:, j] #Get the Entire Column of the IMU data
        ang_vel_col = vals_imu_col[-3:] #Get the angular velocities
        ang_vel_col_new = array([ang_vel_col[1], ang_vel_col[2], ang_vel_col[0]]) #Re-Order the components into Roll Pitch Yaw format


        #Predict orientation at next time step
        quat_next = motion_model(ang_vel_col_new, quat_start, tau)

        Q[j + 1, :] = np.array(quat_next) #Saves quaternions for PGD

        angles_imu[:, j] = log_quat(quat_next)

        #Update
        quat_start = quat_next
        tau = time_imu_new[j + 1] - time_imu_new[j]


    #Get Vicon Data
    for k in range(row1.shape[1] - 1):

        #ith rotation
        Ri = rots[:, :, k]

        #Convert to Euler angles
        rotation = Rotation.from_matrix(Ri)
        euler_angles = rotation.as_euler('xyz')

        angles[:, k] = transpose(euler_angles)



    #Perform Gradient Descent Computation
    next_vec = Q[:] #Current iteration vector
    list_cost = [] #List of the cost function for later plotting
    alpha = 0.01
    for l in range(num_iters):
        deriv = grad(cost_func) #Compute the gradient

        ts = tic()
        deriv_eval = deriv(next_vec) #Evaluate gradient at current iteration
        #Iteration to new vector
        new_vec = next_vec - (alpha * deriv_eval) #Update step

        #Divide by its norm method
        #norms_new_vec = norm(new_vec, ord=2, axis=1)
        #next_vec = new_vec / norms_new_vec[:, np.newaxis]
        next_vec = new_vec / np.repeat(np.expand_dims(np.linalg.norm(new_vec, axis=1), axis=1), 4, axis=1)


        toc(ts, "Computing Gradient")
        updated_cost = cost_func(next_vec)
        print('cost: ', updated_cost)
        list_cost.append(updated_cost)
        print("End of iteration: ", l)

    #Convert optimized quaternions in euler angles
    opt_angles = zeros((3, next_vec.shape[0]))
    for m in range(next_vec.shape[0]):
        opt_angles[:, m] = np.transpose(log_quat(next_vec[m, :]))

    #COMMENT HERE
    #VICON Ground Truth
    y1 = angles[0, :]
    y2 = angles[1, :]
    y3 = angles[2, :]
    #COMMENT HERE


    #Unoptimized motion model
    y11 = angles_imu[0, :]
    y22 = angles_imu[1, :]
    y33 = angles_imu[2, :]

    #Optimized motion model
    y111 = opt_angles[0, :]
    y222 = opt_angles[1, :]
    y333 = opt_angles[2, :]

    #COMMENT HERE
    axes[0].plot(y1, label='Vicon - Ground Truth Roll')
    #COMMENT HERE
    axes[0].plot(y11, label='NOT Optimized IMU Roll')
    axes[0].plot(y111, label='Optimized IMU Roll')
    axes[0].set_title(f'IMU Roll - Vicon Roll - Optimized Roll Trajectory for Data Set {i + 1}')
    axes[0].legend()

    #COMMENT HERE
    axes[1].plot(y2, label='Vicon - Ground Truth Pitch')
    #COMMENT HERE
    axes[1].plot(y22, label='NOT Optimized IMU Pitch')
    axes[1].plot(y222, label='Optimized Pitch Trajectory')
    axes[1].set_title(f'IMU Pitch - Vicon Pitch - Optimized Pitch Trajectory for Data Set {i + 1}')
    axes[1].legend()

    #COMMENT HERE
    axes[2].plot(y3, label='Vicon - Ground Truth Yaw')
    #COMMENT HERE
    axes[2].plot(y33, label=' NOT Optimized IMU Yaw')
    axes[2].plot(y333, label='Optimized Yaw Trajectory')
    axes[2].set_title(f'IMU Yaw - Vicon Yaw - Optimized Yaw Trajectory for Data Set {i + 1}')
    axes[2].legend()

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()

    #Plot cost
    plt.plot(list_cost)
    plt.title(f'Cost Function for Data Set {i + 1}')
    plt.show()





print("Done")