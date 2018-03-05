import cv2
import os
import numpy as np
from PIL import Image

image_folder = 'point_cloud/'
CELLS = 600
ROI = 60

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('video.avi', fourcc, 10.0, (600,600), False)

for i in range(1,10000):
    print(i)
    point_cloud = np.genfromtxt(image_folder + 'pc_%i.csv'%i, delimiter=' ')

    grid = np.zeros([CELLS,CELLS])
    for point in point_cloud:
        x, y, z = point
        x += ROI/2
        x /= ROI
        cell_x = int(x*CELLS -1)

        y += ROI/2
        y /= ROI
        cell_y = int(y*CELLS -1)

        grid[cell_x,cell_y] += 1

    grid = np.uint8(100*grid)
    #img = Image.fromarray(grid)
    #img = img.rotate(180);
    grid = np.rot90(grid,2)

    video.write(grid)

video.release()
cv2.destroyAllWindows()
