function [pixels, angles] = GetRelevantPixels(lidarResH,lidarResV,imgWidth,imgHeight,fovH)

    d = imgWidth/(2*tan(degtorad(fovH)/2));
    % Define three points in the image plane
    p1 = [-imgWidth/2 -d -imgHeight/2];
    p2 = [-imgWidth/2 -d imgHeight/2];
    p3 = [0 -d 0];

    % Calculate the image plane normal
    normal = cross(p1-p2, p1-p3);
    normal = normal/norm(normal);


    %------------------------
    lineP=p2*Rotz(-fovH/2);
    [pixel, check] = plane_line_intersect(normal,p3,lineP,[0 0 0]);
    offset = (imgHeight/2) -pixel(3);
    fakeImgHeight = imgHeight - offset*2;
    %--------------------
    % Calculate focal length and vertical FOV
    fovV = radtodeg(2*atan(fakeImgHeight/(2*d))); %weird

    %
    nAnglesH = floor(fovH/lidarResH);
    nAnglesV = floor(fovV/lidarResV);

    % Set start angles for horizontal and vertical rotation around origo
    startAngleV = -fovV/2;
    startAngleH = -fovH/2;

    % Create line segment to check for intersection with image plane
    lineP1 = [0,0,0];
    lineP=p1*Rotz(-fovH/2);
    [pixel, check] = plane_line_intersect(normal,p3,lineP,[0 0 0]);
    lineP2 = [0, -d, pixel(3)]; % p5
    line = (lineP2 - lineP1);

    angles = zeros(nAnglesV*nAnglesH,2);
    pixels = zeros(nAnglesV*nAnglesH,3);
    
    index = 1;

    figure
    for stepV = 1:nAnglesV
        lineH = line*Rotx((stepV-1)*lidarResV); % Rotate up
        
        angleV = startAngleV + (stepV-1)*lidarResV;
        lineH = lineH*Rotz(-startAngleH); % Rotate to horizontal start position

        for stepH = 1:nAnglesH
            [pixel, check] = plane_line_intersect(normal,p3,lineH,[0 0 0]);
            lineH = lineH*Rotz(-lidarResH);
            angleH = startAngleH + (stepH-1)*lidarResH;
            
            angles(index,:) = [angleV angleH];
            pixels(index,:) = [pixel(1), pixel(2),pixel(3)];
            index = index + 1;
        end
    end
    
    
    %pixels(:,2) = int32(pixels(:,1)) + 1;
    pixels(:,1) = int32(pixels(:,1)) + imgWidth/2 + 1;
    pixels(:,3) = int32(pixels(:,3)) + imgHeight/2 + 1;
    
    %scatter(angles(:,1),angles(:,2));
    scatter3(pixels(:,1),pixels(:,2),pixels(:,3))
    
end

