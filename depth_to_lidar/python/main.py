import numpy as np
from scipy import misc
from depth_to_point_cloud import depth_to_point_cloud
from point_cloud_to_image import trim_to_roi

# Region of interest side length, in meters.
roi = 60
cells = 60

# Read depth maps
head = misc.imread('../example_images/head/image_00001.png')
tail = misc.imread('../example_images/tail/image_00001.png')
left = misc.imread('../example_images/left/image_00001.png')
right = misc.imread('../example_images/right/image_00001.png')

# Convert depth maps to 3D point cloud
point_cloud = depth_to_point_cloud(head, tail, left, right)
# Trim point cloud to only contain points within the region of interest
point_cloud = trim_to_roi(point_cloud,roi)
# TODO Count number of points within each grid cell
# TODO Save as image

np.savetxt("trimmed_point_cloud.csv", point_cloud, delimiter=",")
