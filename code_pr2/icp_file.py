#from utils import read_canonical_model, load_pc

import autograd.numpy as np
#import utils
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import transforms3d as t3d
from autograd import grad
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
import torch
import dgl.geometry


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

def create_rotation_along_z(theta):

    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

# Multiply matrices
# Given matrices T1, T2, ..., Tn multiply Tn*Tn-1* ... *T1
def multiply_matrices(T):

    #Where T is of dimension (4, 4, # of iterations)

    for i in range(T.shape[2] -1):

        if i == 0:
            T_curr = T[:, :, i]

        T_next = T[:, :, i+1]
        T_curr = np.matmul(T_next, T_curr)

    return T_curr



def kabsch(point_cloud_z, point_cloud_m):

    # point_cloud_z is of dimension 3 by N
    # point_cloud_m is of dimension 3 by N



    #Compute the centroids
    cen_m = np.mean(point_cloud_m, axis = 1)
    cen_z = np.mean(point_cloud_z, axis = 1)

    #Copmute the distances of the point clouds to their center
    del_m = point_cloud_m - cen_m.reshape(3, 1)
    del_z = point_cloud_z - cen_z.reshape(3, 1)

    Q = np.zeros((3, 3))
    for k in range(point_cloud_m.shape[1]):
        # Q = Q + (del_m[:, k] * np.transpose(del_z[:, k]))
        Q = Q + np.outer(del_m[:, k], del_z[:, k])



    U, _, Vt = np.linalg.svd(Q) #Get SVD

    M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(U @ Vt)]])
    R_next = U @ M @ Vt # Compute R_next

    #print(np.linalg.det(R_next))

    p_next = cen_m - np.dot(R_next, cen_z)


    return R_next, p_next, np.trace(np.matmul(Q.T, R_next))


# def make_association(source, target):
#
#     # Shapes for source and target are (n, dim) and (m, dim) respectively
#
#     dim = source.shape[1]
#     assert source.shape[1] == dim
#     assert target.shape[1] == dim
#
#     source = source.copy()
#     target = target.copy()
#     flip = False
#     if source.shape[0] < target.shape[0]:
#         flip = True
#         source, target = target, source
#
#     assert source.shape[0] >= target.shape[0]
#
#     # Build Tree for Smallest pt cloud
#     tree = KDTree(source)
#
#     _, delta = tree.query(target)
#
#     z = source[delta]
#
#     if flip:
#         z, target = target, z
#
#     return z, target






def icp(iters, target, source, R_curr, p_curr):


    num_sampled_pts = 5 #Number of points to sample from the cloud

    point_cloud_m = target

    point_cloud_z = source
    #sampled_point_cloud_z = np.transpose(point_cloud_z[np.random.choice(point_cloud_z.shape[0], num_sampled_pts, replace=False)])
    #sampled_point_cloud_z = np.transpose(point_cloud_z[::num_sampled_pts])
    sampled_point_cloud_z = np.transpose(point_cloud_z)


    # Build a tree
    #tree_source = KDTree(point_cloud_z)
    #sampled_point_cloud_z = np.transpose(point_cloud_z)


    T_final = np.zeros((4, 4))
    #T_save = np.zeros((4, 4, iters + 1))
    #loss = np.zeros(iters)
    #print("Begin ICP")



    #T_save[:, :, 0] = create_pose(R_curr, p_curr)
    prev_loss = np.inf
    for j in range(iters):



        # Create the Data Association
        trans_sampled_point_cloud_z = np.transpose(np.matmul(R_curr, sampled_point_cloud_z) + p_curr.reshape(3, 1))
        #trans_sampled_point_cloud_m = np.transpose(R_curr) @ (np.transpose(point_cloud_m) - p_curr.reshape(3, 1))
        #delta = np.argmin(np.sum((trans_sampled_point_cloud_z[:, np.newaxis, :] - point_cloud_m) ** 2, axis = -1), axis = 1) #Source to target
        tree = KDTree(trans_sampled_point_cloud_z)



        #delta = find_nbr(trans_sampled_point_cloud_z, point_cloud_m)
        _, delta = tree.query(point_cloud_m, k = 1)
        loss = np.mean(np.linalg.norm(np.transpose(np.transpose(sampled_point_cloud_z)[delta]) - point_cloud_m.T, axis = 1))

        if abs(prev_loss - loss) < 1e-6:
            break
        prev_loss = loss

        #z, m = make_association(trans_sampled_point_cloud_z, point_cloud_m)

        #Define new point cloud bases on the Data association
        #new_point_cloud_m = np.zeros((3, len(delta)))
        #for k in range(len(delta)):
            #new_point_cloud_m[:, k] = np.transpose(point_cloud_m[delta[k], :])

        #Call Kabsch
        # Which will give us the optimal orientation and translation between the two point clouds

        # sampled_point_cloud_z = sampled_point_cloud_z.T
        #R_next, p_next, _ = kabsch(sampled_point_cloud_z, np.transpose(point_cloud_m[delta]))
        R_next, p_next, _ = kabsch(np.transpose(np.transpose(sampled_point_cloud_z)[delta]), point_cloud_m.T)
        #print(z.shape)
        #print(m.shape)
        #3R_next, p_next, _ = kabsch((R_curr.T @ (z.T - p_curr.reshape(3, 1))), np.transpose(m))

        # Maybe put a stopping point
        #loss[j] = np.trace((((np.dot(R_next, sampled_point_cloud_z) + p_next.reshape(3, 1)) - new_point_cloud_m).T @ ((np.dot(R_next, sampled_point_cloud_z) + p_next.reshape(3, 1)) - new_point_cloud_m)))

        #Update
        #sampled_point_cloud_z = np.transpose(point_cloud_z[np.random.permutation(num_sampled_pts)])
        R_curr = R_next
        p_curr = p_next
        #T_save[:, :, j + 1] = create_pose(R_curr, p_curr)
        #sampled_point_cloud_z = trans_sampled_point_cloud_z.T
        #print("End of iteration", j)



    #Construct the final pose T
    #T_final[0:3, 0:3] = R_curr
    #T_final[0, 3] = p_curr[0]
    #T_final[1, 3] = p_curr[1]
    #T_final[2, 3] = p_curr[2]
    #T_final[3, 3] = 1

    T_final = create_pose(R_curr, p_curr)

    # Save poses matrices at each iteration
    #T_save[:, :, j] = T_next.copy()

    #T = multiply_matrices(T_save)
    #print("Done with ICP")

    return T_final, _


# iters = 100
# source = np.array([[1, 1, 1],
#               [3, 1, 1],
#               [2, 2, 1],
#               [1, 3, 0]])
#
# # source = np.random.rand(1000, 3)
#
# target = np.array([[7, 2, 3],
#               [5, 3, 2],
#               [6, 4, 1],
#               [4, 5, 6],
#               [0, 0, 2]])
#
# # target = np.random.rand(800, 3)
#
# T_initial = create_pose(np.eye(3), np.mean(source.T, axis = 1) - np.mean(target.T, axis = 1))
#
# #utils.visualize_icp_result(source, target, T_initial)
# T_final, _ = icp(iters, target, source, np.eye(3), np.mean(source.T, axis = 1) - np.mean(target.T, axis = 1))
# utils.visualize_icp_result(source, target, T_final)


