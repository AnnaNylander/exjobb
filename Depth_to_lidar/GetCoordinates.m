function [x,y,z] = GetCoordinates(imageA,angles, pixels)
    pixels = uint32(pixels);
    nPoints = length(pixels);
    c = zeros(nPoints,3);
    
    %for i = 1:nVertical
       for i = 1:nPoints
           v = degtorad(angles(i,1));
           h = degtorad(angles(i,2));
           r = imageA(pixels(i,3),pixels(i,1));
           
           [x,y,z] = sph2cart(h,v,r);
           
           %c(pointIndex,1) = (1/cos(h))*x;
           %c(pointIndex,2) = (1/cos(h))*y;
           %c(pointIndex,3) = z;
           
           c(i,1) = -(1/cos(pi - h))*x; %-h
           c(i,2) = (1/sin(pi/2 - h))*y; %pi/2 -h
           c(i,3) = z;
       end
    %end

    x = c(:,1);
    y = c(:,2);
    z = c(:,3);
end

