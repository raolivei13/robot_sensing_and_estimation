
import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result
from icp_file import icp
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import torch
import dgl.geometry

# if __name__ == "__main__":
#   obj_name = 'drill' # drill or liq_container
#   num_pc = 4 # number of point clouds
#
#   source_pc = read_canonical_model(obj_name)
#
#   for i in range(num_pc):
#     target_pc = load_pc(obj_name, i)
#
#     # estimated_pose, you need to estimate the pose with ICP
#     pose = np.eye(4)
#
#     # visualize the estimated result
#     visualize_icp_result(source_pc, target_pc, pose)

def test(obj_name, num_pc, est_pose):

  source_pc = read_canonical_model(obj_name)
  #source_pc = source_pc[::50]

  #sampled_point_cloud_z = source_pc[np.random.permutation(5000)]

  for i in range(num_pc):
    target_pc = load_pc(obj_name, i)

    # estimated_pose, you need to estimate the pose with ICP
    pose = est_pose[:, :, i]
    #pose = est_pose


    # visualize the estimated result
    visualize_icp_result(source_pc, target_pc, pose)



iters = 100
obj_name = 'liq_container'
num_pc = 4


poses = np.zeros((4, 4, 4)) # Final pose for each point cloud

R_initial = np.eye(3)
#loss_vec = np.zeros((4, iters))
# Iterate through the number of point clouds
for i in range(num_pc):


  #R_initial = R_best[:, :, i]
  #R_initial = np.array([[np.cos(yaws[yaw_best_idx[i]]), -np.sin(yaws[yaw_best_idx[i]]), 0], [np.sin(yaws[yaw_best_idx[i]]), np.cos(yaws[yaw_best_idx[i]]), 0], [0, 0, 1]])

  target = load_pc(obj_name, i) #Load target
  source = read_canonical_model(obj_name) #Load source


  p_initial = np.mean(source.T, axis = 1) - np.mean(target.T, axis = 1)
  #p_initial = np.array([0, 0, 0])
  #p_initial = np.array([1, 0, 0])
  #p_initial = np.array([0, 0.5, 0])
  poses[:, :, i], _ = icp(iters, target, source, R_initial, p_initial)


# plt.figure()
# plt.plot(loss_vec[0, :])
# plt.plot(loss_vec[1, :])
# plt.plot(loss_vec[3, :])
# plt.plot(loss_vec[3, :])
# plt.xlabel("Iterations")
# plt.ylabel("Loss")
# plt.show()



test(obj_name, num_pc, poses) #Test and visualize ICP result