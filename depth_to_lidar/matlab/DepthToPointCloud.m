FAR = 1000;           % Far field distance
LIDAR_RES_H = 0.16;   % Horizontal angular resolution of lidar
LIDAR_RES_V = 1.33;   % Vertical angular resolution of lidar
CAMERA_WIDTH = 500;  % Width in pixels of the depth images (2000) (500)
CAMERA_HEIGHT = 420; % Height in pixels of depth images (1677) (420)
FOV_H = 90;           % Horizontal field of view
INTERPOLATE = false;   % If true, depth values are interpolated in query points
THRESHOLD = 2;        % Do not interpolate if depth difference is more than threshold
ROI = 60;             % Omitting points outside a square region of interest with side = ROI

% Decode image data to depth map
head = DecodeDepth(imread('../example_images/head/image_00060.png'),FAR);
tail = DecodeDepth(imread('../example_images/tail/image_00060.png'),FAR);
left = DecodeDepth(imread('../example_images/left/image_00060.png'),FAR);
right = DecodeDepth(imread('../example_images/right/image_00060.png'),FAR);

% Calculate the coordinates hit by lidar rays in the image, together with
% the corresponding lidar ray angles.
[coords, angles] = GetRelevantCoordinates(LIDAR_RES_H,LIDAR_RES_V,CAMERA_WIDTH,CAMERA_HEIGHT, FOV_H);

yCoords = coords(:,2);
zCoords = coords(:,3);

% Concatenate coordinates and angles, and store in csv file for use in Python
data = [coords(:,2), coords(:,3), angles(:,1),angles(:,2)];
csvwrite('coords_and_angles.csv',data);

if INTERPOLATE
    % Get interpolated value at each point
    vHead = Interpolate2D(head, yCoords, zCoords,THRESHOLD);
    vTail = Interpolate2D(tail, yCoords, zCoords,THRESHOLD);
    vLeft = Interpolate2D(left, yCoords, zCoords,THRESHOLD);
    vRight = Interpolate2D(right, yCoords, zCoords,THRESHOLD);
else
    % Return the pixel values that the query points are hitting,
    % without interpolation.
    vHead = GetPixelValues(head, yCoords, zCoords);
    vTail = GetPixelValues(tail, yCoords, zCoords);
    vLeft = GetPixelValues(left, yCoords, zCoords);
    vRight = GetPixelValues(right, yCoords, zCoords);
end

% Calculate coordinates from angles and depth values
[xH,yH,zH] = GetCoordinates(vHead,angles);
[xT,yT,zT] = GetCoordinates(vTail,angles);
[xL,yL,zL] = GetCoordinates(vLeft,angles);
[xR,yR,zR] = GetCoordinates(vRight,angles);

% Rotate coordinates according to which camera they correspond to
[xT,yT,zT] = ZAxisRotation(xT,yT,zT,degtorad(180));
[xL,yL,zL] = ZAxisRotation(xL,yL,zL,degtorad(-90));
[xR,yR,zR] = ZAxisRotation(xR,yR,zR,degtorad(90));

[xH,yH,zH] = TrimToRoi([xH,yH,zH],ROI);
[xT,yT,zT] = TrimToRoi([xT,yT,zT],ROI);
[xL,yL,zL] = TrimToRoi([xL,yL,zL],ROI);
[xR,yR,zR] = TrimToRoi([xR,yR,zR],ROI);

% Plot each camera's point cloud
figure;
scatter3(xT,yT,zT,'.');
hold on
scatter3(xH,yH,zH,'.');
hold on
scatter3(xL,yL,zL,'.');
hold on
scatter3(xR,yR,zR,'.');
set(gca,'Color','k');
daspect([1 1 1]);
