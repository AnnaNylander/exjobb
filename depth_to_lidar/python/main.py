import numpy as np
from scipy import misc
from depth_to_point_cloud import depth_to_point_cloud
from point_cloud_to_image import trim_to_roi
from PIL import Image

# Region of interest side length, in meters.
roi = 60
cells = 600

# Read depth maps
head = misc.imread('../example_images/head/image_00034.png')
tail = misc.imread('../example_images/tail/image_00034.png')
left = misc.imread('../example_images/left/image_00034.png')
right = misc.imread('../example_images/right/image_00034.png')

# Convert depth maps to 3D point cloud
point_cloud = depth_to_point_cloud(head, tail, left, right)
# Trim point cloud to only contain points within the region of interest
point_cloud = trim_to_roi(point_cloud,roi)
# TODO Count number of points within each grid cell
# TODO Save as image

grid = np.zeros([cells,cells])
print(point_cloud.shape)
for point in point_cloud:
    x, y, z = point
    x += roi/2
    x /= roi
    cell_x = int(x*cells)

    y += roi/2
    y /= roi
    cell_y = int(y*cells)

    grid[cell_x,cell_y] += 1

grid = 64*grid

img = Image.fromarray(grid)
img = img.rotate(180);
#img.save('my.png')
img.show()
#np.savetxt("pointCloudForTestInMatLab.csv", point_cloud, delimiter=",")
