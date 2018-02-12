function [pixels, angles] = GetRelevantPixels(lidarResH,lidarResV,imgWidth,imgHeight,fovH)
    % TODO Use these more specific vertical angles
    velodyneAngles = [-30.67;-9.33;-29.33;-8.00;-28.00;-6.66;-26.66;-5.33; 
                        -25.33;-4.00;-24.00;-2.67;-22.67;-1.33;-21.33;0.00; 
                        -20.00;1.33;-18.67;2.67;-17.33;4.00;-16.00;5.33;
                        -14.67;6.67;-13.33;8.00;-12.00;9.33;-10.67;10.67]';

    % Calculate distance from camera point to near plane center point 
    d = imgWidth/(2*tan(degtorad(fovH)/2));
    
    % Define three points in the image plane
    p1 = [d imgWidth/2 -imgHeight/2];
    p2 = [d imgWidth/2 imgHeight/2];
    p3 = [d 0 0];

    % Calculate the image plane normal
    normal = cross(p1-p2, p1-p3);
    normal = normal/norm(normal);

    % Calculate offset from bottom of depth map to 
    lineP = p1*Rotz(fovH/2);
    [pixel, check] = plane_line_intersect(normal,p3,lineP,[0 0 0]);
    offset = (imgHeight/2) + pixel(3);
    fakeImgHeight = imgHeight - offset*2;
    
    % Calculate focal length and vertical FOV
    fovV = radtodeg(2*atan(fakeImgHeight/(2*d))); %weird

    % Calculate number of 
    nAnglesH = floor(fovH/lidarResH) + 1;
    nAnglesV = 32;

    % Set start angles for horizontal and vertical rotation around origo
    startAngleV = -fovV/2;
    startAngleH = fovH/2;

    % Create line segment to check for intersection with image plane
    [pixel, check] = plane_line_intersect(normal,p3,lineP,[0 0 0]);
    lineP2 = [d, 0, pixel(3)]; % p5
    line = (lineP2 - [0 0 0]);

    angles = zeros(nAnglesV*nAnglesH,2);
    pixels = zeros(nAnglesV*nAnglesH,3);
    
    index = 1;

    % Loop over all lidar ray angles and check intersecting coordinates in
    % near plane.
    for stepV = 1:nAnglesV
        lineH = line*Roty((stepV-1)*lidarResV); % Rotate up
        
        angleV = startAngleV + (stepV-1)*lidarResV;
        lineH = lineH*Rotz(-startAngleH); % Rotate to horizontal start position

        for stepH = 1:nAnglesH
            [pixel, check] = plane_line_intersect(normal,p3,lineH,[0 0 0]);
            lineH = lineH*Rotz(lidarResH);
            angleH = startAngleH - (stepH-1)*lidarResH;
            
            angles(index,:) = [angleV angleH];
            pixels(index,:) = [pixel(1), pixel(2),pixel(3)];
            index = index + 1;
        end
    end
    
    % Shift coordinates to form actual pixel indices
    pixels(:,2) = int32(imgWidth - (pixels(:,2) + imgWidth/2) + 1);
    pixels(:,3) = int32(pixels(:,3) + imgHeight/2 + 1);
    
    % Plot intersection points between lidar rays and near plane
    scatter3(pixels(:,1),pixels(:,2),pixels(:,3),'.');
    
end

