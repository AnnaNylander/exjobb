far = 1000;             % Far field distance
lidarResH = 0.16;       % Horizontal angular resolution of lidar
lidarResV = 1.33;       % Vertical angular resolution of lidar
cameraWidth = 2000;     % Width in pixels of the depth images
cameraHeight = 1677;    % Height in pixels of depth images
fovH = 90;              % Horizontal field of view

% Decode image data to depth map
head = DecodeDepth(imread('../example_images/head/image_00001.png'),far);
tail = DecodeDepth(imread('../example_images/tail/image_00001.png'),far);
left = DecodeDepth(imread('../example_images/left/image_00001.png'),far);
right = DecodeDepth(imread('../example_images/right/image_00001.png'),far);

% Calculate which pixels that are hit by lidar rays. Also, return lidar ray
% angles.
[pixels, angles] = GetRelevantPixels(lidarResH,lidarResV,cameraWidth,cameraHeight,fovH);

% Concatenate pixels and angles, and store in csv file for use in Python
data = [pixels(:,2), pixels(:,3), angles(:,1),angles(:,2)];
csvwrite('pixels_and_angles.csv',data);

% Calculate coordinates from angles and depth values
[xH,yH,zH] = GetCoordinates(head,angles,pixels);
[xT,yT,zT] = GetCoordinates(tail,angles,pixels);
[xL,yL,zL] = GetCoordinates(left,angles,pixels);
[xR,yR,zR] = GetCoordinates(right,angles,pixels);

% Rotate coordinates according to which camera they correspond to
[xT,yT,zT] = ZAxisRotation(xT,yT,zT,degtorad(180));
[xL,yL,zL] = ZAxisRotation(xL,yL,zL,degtorad(-90));
[xR,yR,zR] = ZAxisRotation(xR,yR,zR,degtorad(90));

% Stack all coordinates
x = [xH;xT;xL;xR];
y = [yH;yT;yL;yR];
z = [zH;zT;zL;zR];

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
