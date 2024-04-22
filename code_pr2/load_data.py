import numpy as np




def get_data(dataset):

    with np.load("data/Encoders%d.npz" % dataset) as data:
      encoder_counts = data["counts"]  # 4 x n encoder counts
      encoder_stamps = data["time_stamps"]  # encoder time stamps

    with np.load("data/Hokuyo%d.npz" % dataset) as data:
      lidar_angle_min = data["angle_min"]  # start angle of the scan [rad]
      lidar_angle_max = data["angle_max"]  # end angle of the scan [rad]
      lidar_angle_increment = data["angle_increment"]  # angular distance between measurements [rad]
      lidar_range_min = data["range_min"]  # minimum range value [m]
      lidar_range_max = data["range_max"]  # maximum range value [m]
      lidar_ranges = data["ranges"]  # range data [m] (Note: values < range_min or > range_max should be discarded)
      lidar_stamsp = data["time_stamps"]  # acquisition times of the lidar scans

    with np.load("data/Imu%d.npz" % dataset) as data:
      imu_angular_velocity = data["angular_velocity"]  # angular velocity in rad/sec
      imu_linear_acceleration = data["linear_acceleration"]  # accelerations in gs (gravity acceleration scaling)
      imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

    with np.load("data/Kinect%d.npz" % dataset) as data:
      disp_stamps = data["disparity_time_stamps"]  # acquisition times of the disparity images
      rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

    data_encoder = [encoder_counts, encoder_stamps]
    data_hok = [lidar_angle_min, lidar_angle_max, lidar_angle_increment, lidar_range_min, lidar_range_max, lidar_ranges, lidar_stamsp]
    data_imu = [imu_angular_velocity, imu_linear_acceleration, imu_stamps]
    data_kin = [disp_stamps, rgb_stamps]


    return data_encoder, data_hok, data_imu, data_kin
