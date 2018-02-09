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
           
           %final correction.
           c(i,1) = (1/cos(h))*(1/cos(v))*x;
           c(i,2) = (1/cos(h))*(1/cos(v))*y;
           c(i,3) = (1/cos(h))*(1/cos(v))*z;
       end
    %end

    x = c(:,1);
    y = c(:,2);
    z = c(:,3);
end
