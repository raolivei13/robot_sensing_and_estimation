
Hello,
attached are instructions on how to run Project 2: LiDAR - Based SLAM.
All of the code has been written in PyCharm.

Instructions:
- install the dependencies from "requirements.txt"
- have ALL .py files in the same directory: i.e load_data.py, part_one.py, occ_grid.py, scan_matching.py, text_mapping.py including the data sets: data, dataRGBD, icp_warm_up
  - EXCEPT: for icp_file.py and test_icp.py, these are whithin the directory icp_warm_up

For part_one.py run the file and get results for dataset 20 and 21 on same graph
test_icp.py runs icp_file.py and uses utils.py for point cloud visualization

For scan_matching.py, on top of the file modify dataset parameter for either 20 or 21
the file will return the optimized trajectory with LiDAR based measurements.
After this, run occ_grid.py, text_mapping.py, gtsam_part.py while modifying also the parameter dataset
to be either 20 and 21, if scan_matching.py was done for dataset = 20 then the same should be
done for all the other files.


IMPORTANT: The files submitted are not in this directory structure, make sure that when running the code you have the structure as listed below. 

Directory Structure:
  - data (Here you have Encoder, Hokuyo, IMU, Kinect DataSets for 20 and 21)
  - dataRGBD (Here you have RGBD & Disparity DataSets for 20 and 21)
  - icp_warm_up
    - data (drill and liquid container folder)
      - drill
      - liq_container
    - icp_file.py
    - test_icp.py
    - utils.py
  - load_data.py
  - part_one.py
  - scan_matching.py
  - occ_grid.py
  - text_mapping.py
  - gtsam_part.py
