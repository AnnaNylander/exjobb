import numpy as np
from scipy import misc

def rotate_z(points,theta):
    """Rotate an array of points theta degrees around the z-axis."""
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    rotationMatrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.dot(points,rotationMatrix)

def depth_to_point_cloud(imgHead, imgTail, imgLeft, imgRight, far, interpolate=False, threshold=1.0):
    """Returns a point cloud calculated from four depth map."""
    # Read depth images and decode them
    head = decode_depthmap(imgHead, far)
    tail = decode_depthmap(imgTail, far)
    left = decode_depthmap(imgLeft, far)
    right = decode_depthmap(imgRight, far)

    # Get list of lidar ray angles and the coordinates in the image they intersect
    [coords, angles] = get_relevant_coordinates()
    x_coords, y_coords = coords[:,0], coords[:,1]

    n_points = len(x_coords)
    vHead = np.zeros([n_points,1])
    vTail = np.zeros([n_points,1])
    vLeft = np.zeros([n_points,1])
    vRight = np.zeros([n_points,1])

    if interpolate:
        vHead = interpolate_2D(head, x_coords, y_coords, threshold)
        vTail = interpolate_2D(tail, x_coords, y_coords, threshold)
        vLeft = interpolate_2D(left, x_coords, y_coords, threshold)
        vRight = interpolate_2D(right, x_coords, y_coords, threshold)
    else:
        vHead = get_pixel_values(head, x_coords, y_coords)
        vTail = get_pixel_values(tail, x_coords, y_coords)
        vLeft = get_pixel_values(left, x_coords, y_coords)
        vRight = get_pixel_values(right, x_coords, y_coords)

    # Calculate 3D coordinates from ray angles and depth values
    cHead = get_coordinates(vHead,angles)
    cTail = get_coordinates(vTail,angles)
    cLeft = get_coordinates(vLeft,angles)
    cRight = get_coordinates(vRight,angles)

    # Rotate points according to the camera directions
    cTail = rotate_z(cTail,180)
    cLeft = rotate_z(cLeft,-90)
    cRight = rotate_z(cRight,90)

    # Concatenate points from all cameras and save as image
    pointCloud = np.concatenate((cHead,cTail,cLeft,cRight),0)
    return pointCloud

def interpolate_2D(values,x_query,y_query,threshold):
    """Conditioned interpolation between pixel values in the queried points.

    Let q be the pixel that the query point lies in. If the difference in pixel
    value between q and the three neighboring pixels closest to the query point
    is larger than threshold, then no interpolation is performed; the value of q
    is used."""

    # Four pixels with indices cX and cY:
    # (cX1,cY1) (cX2,cY1)
    # (cX1,cY2) (cX2,cY2)

    n_points = len(x_query)
    interpolatedValues = np.zeros([n_points,1]);

    # For each point
    for i in range(n_points):

        # Instantiate coordinate variables
        cX1, cX2, cY1, cY2 = 0, 0, 0, 0
        fx, fy = 0, 0

        # Get the query point coordinates
        Xq = x_query[i] - 1
        Yq = y_query[i] - 1

        # Get horizontal pixel index on which the point lies
        cX = Xq.astype(int)

        # decimal part of point from pixel's beginning
        decimalX = Xq - cX

        # If the queried point is to the left of the pixel's center
        if decimalX <= 0.5:
            cX1 = cX # Set cX1 to the be the pixel index of cX
            cX2 = cX - 1 # We're interested in the pixel to the left of cX
            fx = decimalX + 0.5
        else:
            cX1 = cX + 1 # % We're interested in the pixel to the right of cX
            cX2 = cX # Set cX2 to the be the pixel index of cX
            fx = decimalX - 0.5

        # Repeat procedure for y-axis
        cY = Yq.astype(int)
        decimalY = Yq - cY

        if decimalY <= 0.5:
            cY1 = cY
            cY2 = cY - 1
            fy = decimalY + 0.5
        else:
            cY1 = cY + 1
            cY2 = cY
            fy = decimalY - 0.5

        # Get value in each pixel of relevance
        v1, v2, v3, v4 = values[cY1,cX1], values[cY1,cX2], values[cY2,cX1], values[cY2,cX2]

        # Calculate difference in pixel intensity between pixel hit by
        # query point and the other three pixels.
       	v = values[cY,cX]
        d1, d2, d3, d4 = abs(v - v1), abs(v - v2), abs(v - v3), abs(v - v4)

        c = 0
        #If difference is larger than threshold, do not interpolate.
        if max(d1, d2, d3, d4) > threshold:
            c = v
        else:
            # Linearly interpolate between pixel values to find value in query point
            a = v1*fx + v2*(1-fx)
            b = v3*fx + v4*(1-fx)
            c = a*fy + b*(1-fy)

        interpolatedValues[i] = c
    return interpolatedValues

def get_pixel_values(values, x_coords, y_coords):
    x_coords = x_coords.astype(int) - 1 # Compensate for matlab indexing
    y_coords = y_coords.astype(int) - 1

    n_points = len(x_coords)
    print(np.max(y_coords))
    pixel_values = np.zeros([n_points, 1])

    for i in range(n_points):
        pixel_values[i] = values[y_coords[i], x_coords[i]]

    return pixel_values

def get_relevant_coordinates():
    """Returns a numpy ndarray specifying the pixel a lidar ray hits when shot
    through the near plane."""
    coords_and_angles = np.genfromtxt('coords_and_angles.csv', delimiter=',')
    return np.hsplit(coords_and_angles,2)

def decode_depthmap(depth_map, far_plane_distance):
    """Decode CARLA-encoded depth values into meters."""
    depth_map = depth_map[:,:,0] + 256*depth_map[:,:,1] + (256*256)*depth_map[:,:,2];
    depth_map = depth_map / (256*256*256-1);
    depth_map = depth_map * far_plane_distance;
    return depth_map

def spherical_to_cartesian(azimuth, elevation, radius):
    """Convert spherical coordinates into cartesian x,y and z coordinates"""
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    return np.array([x,y,z])

def get_coordinates(values, angles):
    """Convert depth map values into corresponding lidar measurements."""
    coordinates = np.zeros((len(angles),3))

    for i in range(len(angles)-1):
           v = np.radians(angles[i,0])
           h = -np.radians(angles[i,1])
           r = values[i]

           # Correction for differences between lidar and depthmaps measurements
           # TODO this might as well be done in advance
           correctionConstant = 1 / (np.cos(h) * np.cos(v))
           r = r * correctionConstant

           # Calculate cartesian coordinates
           coordinates[i,:] = np.transpose(spherical_to_cartesian(h,v,r))

    return coordinates

if __name__ == "__main__":
    main()
