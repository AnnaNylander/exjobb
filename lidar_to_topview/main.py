import numpy as np
from PIL import Image

ELEVATION_MAX = 6
ELEVATION_MIN = -18

#point_cloud = np.loadtxt('data/hej.csv', delimiter=' ')
#point_cloud = trim_to_roi(point_cloud,ROI)
def main():
    filaname700k = '/home/annaochjacob/Documents/MATLAB/ply_test/000017.csv'
    #filename100k = '/home/annaochjacob/Documents/MATLAB/ply_test/000006.ply'
    point_cloud = np.loadtxt(filaname700k, delimiter=',', skiprows=1)

    point_cloud = trim_to_roi(point_cloud, 60)
    print(np.max(point_cloud))

    # Save maximum elevation
    data_max_elevation = get_max_elevation(1, point_cloud)
    #filename = (PATH_INPUT + 'topviews/max_elevation/me_%i.csv') %frame
    data_max_elevation = np.squeeze(data_max_elevation, 2)
    img = Image.fromarray(data_max_elevation)

    img.show()

    #np.savetxt(filename, data_max_elevation, delimiter=DELIMITER, \
    #    comments=COMMENTS, fmt='%u')

def trim_to_roi(point_cloud,roi):
    """ Remove points outside ROI."""
    inside_roi = np.max(np.absolute(point_cloud), axis=1) < roi/2
    return point_cloud[inside_roi]

def lidar_to_topview(point_cloud, ROI, CELLS):
    grid = np.zeros([CELLS,CELLS,1])

    for point in point_cloud:
        x, y, z = point
        x += ROI/2
        x /= ROI
        cell_x = (int) (x*CELLS) - 1

        y += ROI/2
        y /= ROI
        cell_y = (int) (y*CELLS) - 1

        grid[cell_x,cell_y] += 1

    grid = np.rot90(grid,2)
    return np.uint8(100*grid)

def get_max_elevation(frame, point_cloud, ROI = 60, CELLS = 600):
    grid = np.full([CELLS,CELLS,1], np.nan)

    for point in point_cloud:
        x, y, z = point
        x += ROI/2
        x /= ROI
        cell_x = (int) (x*CELLS) - 1

        y += ROI/2
        y /= ROI
        cell_y = (int) (y*CELLS) - 1

        element = grid[cell_x,cell_y]
        if np.isnan(element) or element < z:
             grid[cell_x,cell_y] = z

    grid = np.nan_to_num(grid) # Replace all nans with zeros
    grid = grid/(ELEVATION_MAX - ELEVATION_MIN)
    grid = grid + 0.5
    grid = np.rot90(grid,1)
    return np.uint8(grid*255)
    #return np.uint8(100*grid) #enhance contrast
    #img = Image.fromarray(grid)
    #img = img.rotate(180)
    #img.show()
    #img.save('output/topview_%i.png' %frame)

if __name__ == "__main__":
    main()
