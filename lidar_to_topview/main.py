import numpy as np
from PIL import Image

#point_cloud = np.loadtxt('data/hej.csv', delimiter=' ')
#point_cloud = trim_to_roi(point_cloud,ROI)

def lidar_to_topview(frame, point_cloud, ROI, CELLS):
    grid = np.zeros([CELLS,CELLS,1])

    for point in point_cloud:
        x, y, z = point
        x += ROI/2
        x /= ROI
        cell_x = (int) (x*CELLS)

        y += ROI/2
        y /= ROI
        cell_y = (int) (y*CELLS)

        grid[cell_x,cell_y] += 1

    return np.uint8(100*grid)


def get_max_elevation(frame, point_cloud, ROI, CELLS):
    grid = np.full([CELLS,CELLS,1], np.nan)

    for point in point_cloud:
        x, y, z = point
        x += ROI/2
        x /= ROI
        cell_x = (int) (x*CELLS)

        y += ROI/2
        y /= ROI
        cell_y = (int) (y*CELLS)

        element = grid[cell_x,cell_y]
        if np.isnan(element) or element < z:
             grid[cell_x,cell_y] = z

    grid = np.nan_to_num(grid) # Replace all nans with zeros
    grid = grid/(np.max(grid) - np.min(grid))
    grid = grid + 0.5
    return np.uint8(grid*255)

    #return np.uint8(100*grid) #enhance contrast
    #img = Image.fromarray(grid)
    #img = img.rotate(180)
    #img.show()
    #img.save('output/topview_%i.png' %frame)

if __name__ == "__main__":
    main()