function [pixelsV, pixelsH, anglesV, anglesH] = GetRelevantPixels(lidarResH,lidarResV,imgWidth,imgHeight,fovH)
    
    % Ensure we don't get larger indices in the output than there are
    % pixels in the image.
    imgWidth = imgWidth - 1;
    imgHeight = imgHeight - 1;

    centerDistance = imgWidth/(2*tan(fovH/2));
    fovV = 2*atan(imgHeight/(2*centerDistance));
    
    % Horizontal pixels
    anglesH = -fovH/2:lidarResH:fovH/2;
    pixelsH = tan(anglesH)*centerDistance;
    pixelsH = uint32(pixelsH + imgWidth/2) + 1;
    
    % Vertical pixels
    anglesV = -fovV/2:lidarResV:fovV/2;
    pixelsV = tan(anglesV)*centerDistance;
    pixelsV = uint32(pixelsV + imgHeight/2) + 1;
    
end

