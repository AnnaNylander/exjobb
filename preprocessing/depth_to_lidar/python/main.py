import numpy as np
from scipy import misc
from depth_to_point_cloud import depth_to_point_cloud
from point_cloud_to_image import trim_to_roi
from PIL import Image

ROI = 60        # Region of interest side length, in meters.
CELLS = 600     # Number of cells on a side in the output image
FAR = 1000      # Far plane distance
THRESHOLD = 1.5

# Read depth maps
head = misc.imread('../example_images/head/image_00060.png')
tail = misc.imread('../example_images/tail/image_00060.png')
left = misc.imread('../example_images/left/image_00060.png')
right = misc.imread('../example_images/right/image_00060.png')

# Convert depth maps to 3D point cloud
point_cloud = depth_to_point_cloud(head, tail, left, right, FAR, interpolate=True, threshold=THRESHOLD)
# Trim point cloud to only contain points within the region of interest
point_cloud = trim_to_roi(point_cloud,ROI)
# TODO Count number of points within each grid cell
# TODO Save as image

grid = np.zeros([CELLS,CELLS])

for point in point_cloud:
    x, y, z = point
    x += ROI/2
    x /= ROI
    cell_x = int(x*CELLS)

    y += ROI/2
    y /= ROI
    cell_y = int(y*CELLS)

    grid[cell_x,cell_y] += 1

grid = 64*grid

img = Image.fromarray(grid)
img = img.rotate(180);
#img.save('my.png')
img.show()
#np.savetxt("pointCloudForTestInMatLab.csv", point_cloud, delimiter=",")
