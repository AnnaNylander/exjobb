import numpy as np
from PIL import Image

CELLS = 600
ROI = 60

point_cloud = np.genfromtxt('pc_5555.csv', delimiter=' ')

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
