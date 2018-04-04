% Calculates coordinates _IN THE IMAGE_ where lidar rays intersect it.
% Each angle in the output corresponds to a coordinate in the image, where
% - the image has (0,0) in the top left corner,
% - the image's Y-axis has positive direction down
% - the image's X-axis has positive direction right
% These axes and directions are not to be confused with those used
% internally in this function, which use the right hand-rule, such that
% (for a right hand):
% - the thumb points in the positive direction of the X-axis
% - the index finger points in the positive direction of the Y-axis
% - the middle finger points in the positive direction of the Z-axis
% and the center of the image is assumed to lie in [0, d, 0], where d is
% the focal length of the camera.
function [x,y, angles] = GetRelevantCoordinates(lidarResH,lidarResV,imgWidth,imgHeight,fovH)
    % TODO Use these more specific vertical angles
    velodyneAngles = [-30.67;-9.33;-29.33;-8.00;-28.00;-6.66;-26.66;-5.33; 
                        -25.33;-4.00;-24.00;-2.67;-22.67;-1.33;-21.33;0.00; 
                        -20.00;1.33;-18.67;2.67;-17.33;4.00;-16.00;5.33;
                        -14.67;6.67;-13.33;8.00;-12.00;9.33;-10.67;10.67]';

    % Calculate distance from camera point to near plane center point 
    d = imgWidth/(2*tan(degtorad(fovH)/2));
    
    % Define three points in the image plane. Dimension order is [x y z].
    p1 = [-imgWidth/2 d -imgHeight/2];
    p2 = [-imgWidth/2 d imgHeight/2];
    p3 = [0 d 0];

    % Calculate the image plane normal
    normal = cross(p1-p2, p1-p3);
    normal = normal/norm(normal);

    % Calculate offset from bottom of depth map to the z-coordinate of the
    % lowest lidar ray where x = 0, i.e. to p5 in the drawings.
    lineP = p1*Rotz(fovH/2);
    [p5, check] = plane_line_intersect(normal,p3,lineP,[0 0 0]);
    offset = (imgHeight/2) + p5(3);
    
    % Calculate the distance between the lowest and highest lidar rays
    % where x = 0.
    fakeImgHeight = imgHeight - offset*2;
    
    % Calculate focal length and vertical FOV required to end up in the
    % corners of the image when rotating a lidar ray around the z-axis.
    fovV = radtodeg(2*atan(fakeImgHeight/(2*d)));

    % Calculate number of angles step to cover the horizontal FOV
    nAnglesH = floor(fovH/lidarResH) + 1;
    % The number of vertical rays is already determined.
    nAnglesV = 32;

    % Set start angles for horizontal and vertical rotation around origo.
    startAngleV = -fovV/2;
    startAngleH = fovH/2;

    % Create line segment to check for intersection with image plane. 
    % This can be thought of as one lidar ray.
    lineP2 = [0, d, p5(3)]; % p5
    line = (lineP2 - [0 0 0]);

    angles = zeros(nAnglesV*nAnglesH,2);
    coordinates = zeros(nAnglesV*nAnglesH,3);
    
    index = 1;

    % Loop over all lidar ray angles and check intersecting coordinates in
    % near plane.
    for stepV = 1:nAnglesV
        % Set the vertical angle for this horizontal sweep
        angleV = startAngleV + (stepV-1)*lidarResV;
        
        % Rotate ray upwards around x-axis. Negative angle as parameter
        % value for Rotx means rotation upwards in coordinate system.
        ray = line*Rotx(-(stepV-1)*lidarResV);
        
        % Rotate to horizontal start position (intersecting p1)
        ray = ray*Rotz(-startAngleH);

        % For each angle step in the horizontal FOV
        for stepH = 1:nAnglesH
            % Find image plane intersection point
            [coord, check] = plane_line_intersect(normal,p3,ray,[0 0 0]);
            ray = ray*Rotz(lidarResH);
            angleH = startAngleH - (stepH-1)*lidarResH;
            
            angles(index,:) = [angleV angleH];
            coordinates(index,:) = [coord(1), coord(2),coord(3)];
            index = index + 1;
        end
    end
    
    x = coordinates(:,1) + imgWidth/2;
    y = imgHeight - (coordinates(:,3) + imgHeight/2);
    
    % Uncomment to plot coordinates
    % scatter3(coordinates(:,1),coordinates(:,2),coordinates(:,3),'.');
    
end

