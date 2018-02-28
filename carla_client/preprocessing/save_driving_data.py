import numpy as np
from preprocessing.depth_to_point_cloud import depth_to_point_cloud
from preprocessing.point_cloud_to_image import trim_to_roi

ROI = 60        # Region of interest side length, in meters.
FAR = 1000      # Far plane distance
THRESHOLD = 1.5 # Do not interpolate if difference between pixels is larger than this
INTERPOLATE = True

def save_driving_data(measurements, sensor_data, frame):
    #measurement_values = np.zeros([n_frames,8])

    head = sensor_data['CameraDepthHead'].data
    tail = sensor_data['CameraDepthTail'].data
    left = sensor_data['CameraDepthLeft'].data
    right = sensor_data['CameraDepthRight'].data

    # Convert depth maps to 3D point cloud
    point_cloud = depth_to_point_cloud(head, tail, left, right, FAR, interpolate=INTERPOLATE, threshold=THRESHOLD)
    # Trim point cloud to only contain points within the region of interest
    point_cloud = trim_to_roi(point_cloud,ROI)
    # Save point cloud for this frame
    np.savetxt("recorded_data/point_cloud/pc_%i.csv" %frame, point_cloud)

    player_measurements = measurements.player_measurements

    measurement_values[frame-1,0] = player_measurements.transform.location.x / 100 #  (cm -> m )
    measurement_values[frame-1,1] = player_measurements.transform.location.y / 100
    measurement_values[frame-1,2] = player_measurements.forward_speed
    measurement_values[frame-1,3] = player_measurements.collision_vehicles
    measurement_values[frame-1,4] = player_measurements.collision_pedestrians
    measurement_values[frame-1,5] = player_measurements.collision_other
    measurement_values[frame-1,6] = 100 * player_measurements.intersection_otherlane
    measurement_values[frame-1,7] = 100 * player_measurements.intersection_offroad
    #measurement_values[frame,0] = number_of_agents

    # Append measurement values to this episode's csv file
    #np.savetxt("recorded_data/measurements/m_%i.csv" %frame, measurement_values)
