function [x,y,z] = GetCoordinates(imageA,angles, pixels)
    pixels = uint32(pixels);
    nPoints = length(pixels);
    c = zeros(nPoints,3);
    
    %for i = 1:nVertical
       for i = 1:nPoints
           v = angles(i,1);
           h = angles(i,2);
           r = imageA(pixels(i,3),pixels(i,1));
           
           [x,y,z] = sph2cart(h,v,r);
           
           %c(pointIndex,1) = (1/cos(h))*x;
           %c(pointIndex,2) = (1/cos(h))*y;
           %c(pointIndex,3) = z;
           
           c(i,1) = x;
           c(i,2) = y;
           c(i,3) = z;
       end
    %end

    x = c(:,1);
    y = c(:,2);
    z = c(:,3);
end

