% Camera and lidar definitions
imgWidth = 2000;
imgHeight = 1000;
resH = 0.32;
resV = 2.66;
fovH = 90;
d = imgWidth/(2*tan(degtorad(fovH)/2));
fovV = radtodeg(2*atan(imgHeight/(2*d)));

% Define three points in the image plane
p1 = [-imgWidth/2 -d -imgHeight/2];
p2 = [-imgWidth/2 -d imgHeight/2];
p3 = [0 -d 0];

% Calculate the image plane normal
normal = cross(p1-p2, p1-p3);
normal = normal/norm(normal);

%
nAnglesH = floor(fovH/resH);
nAnglesV = floor(fovV/resV);

% Set start angles for horizontal and vertical rotation around origo
startAngleV = -fovV/2;
startAngleH = -fovH/2;

intersections = zeros(3,nAnglesH,nAnglesV);
lineP1 = [0,0,0];
lineP2 = [0, -d, -imgHeight/2];%p1;
line = (lineP2 - lineP1);

pixels = zeros(nAnglesH*nAnglesV,3);
anglesV = zeros(nAnglesH*nAnglesV);
anglesH = zeros(nAnglesH*nAnglesV);

for stepV = 1:nAnglesV
    lineH = line*Rotx((stepV-1)*resV); % Rotate up
    lineH = lineH*Rotz(-startAngleH); % Rotate to horizontal start position
    angleV = startAngleV + (stepV-1)*resV;
    
    for stepH = 1:nAnglesH
        %[intersections(:,stepH, stepV)] = plane_line_intersect(normal,p3,lineH,[0 0 0]);
        [intersections(:,stepH, stepV)] = plane_line_intersect(normal,p3,lineH,[0 0 0]);
        lineH = lineH*Rotz(-resH);
        angleH = startAngleH + (stepH-1)*resH;
        index = (stepV-1)*stepH + stepH;
        pixels(index,:) = intersections(:,stepH, stepV);
        anglesV(index) = angleV;
        anglesH(index) = angleH;
    end
    %scatter3(intersections(1,:,stepV),intersections(2,:,stepV),intersections(3,:,stepV));
    %hold on
end




%scatter3(intersections(1,1,:),intersections(2,1,:),intersections(3,1,:));
%scatter3(lines(1,:),lines(2,:),lines(3,:));
%scatter3(hej(:,1),hej(:,2),hej(:,3));


