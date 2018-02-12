import numpy as np
from scipy import misc

# Distance to far plane
far = 1000

def rotate_z(points,theta):
    """Rotate an array of points theta degrees around the z-axis."""
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    rotationMatrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.dot(points,rotationMatrix)

def depth_to_point_cloud(imgHead, imgTail, imgLeft, imgRight):
    """Returns a point cloud calculated from four depth maps"""
    # Read depth images and decode them
    head = decode_depthmap(imgHead, far)
    tail = decode_depthmap(imgTail, far)
    left = decode_depthmap(imgLeft, far)
    right = decode_depthmap(imgRight, far)

    # Read list of lidar ray angles and the pixels they intersect
    [pixels, angles] = get_relevant_pixels()

    # Calculate 3D coordinates from ray angles and depth values
    cHead = get_coordinates(head,angles,pixels)
    cTail = get_coordinates(tail,angles,pixels)
    cLeft = get_coordinates(left,angles,pixels)
    cRight = get_coordinates(right,angles,pixels)

    # Rotate points according to the camera directions
    cTail = rotate_z(cTail,180)
    cLeft = rotate_z(cLeft,90)
    cRight = rotate_z(cRight,-90)

    # Concatenate points from all cameras and save as image
    pointCloud = np.concatenate((cHead,cTail,cLeft,cRight),0)
    return pointCloud

def get_relevant_pixels():
    """Returns a numpy ndarray specifying the pixel a lidar ray hits when shot
    through the near plane."""
    pixels_and_angles = np.genfromtxt('pixels_and_angles.csv', delimiter=',')
    return np.hsplit(pixels_and_angles,2)

def decode_depthmap(depthmap, farPlaneDistance):
    """Decode CARLA-encoded depth values into meters."""
    depthmap = depthmap[:,:,0] + 256*depthmap[:,:,1] + (256*256)*depthmap[:,:,2];
    depthmap = depthmap / (256*256*256-1);
    depthmap = depthmap * farPlaneDistance;
    return depthmap

def spherical_to_cartesian(azimuth, elevation, radius):
    """Convert spherical coordinates into cartesian x,y and z coordinates"""
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    return np.array([x,y,z])

def get_coordinates(image, angles, pixels):
    """Convert depth map values into corresponding lidar measurements."""
    coordinates = np.zeros((len(angles),3))

    for i in range(len(angles)):
           v = -np.radians(angles[i,0])
           h = -np.radians(angles[i,1])
           pixelX = int(pixels[i,0])
           pixelY = int(pixels[i,1])
           r = image[pixelY,pixelX]

           # Correction for differences between lidar and depthmaps measurements
           # TODO this might as well be done in advance
           correctionConstant = 1 / (np.cos(h) * np.cos(v))
           r = r * correctionConstant

           # Calculate cartesian coordinates
           coordinates[i,:] = spherical_to_cartesian(h,v,r)

    return coordinates

if __name__ == "__main__":
    main()
