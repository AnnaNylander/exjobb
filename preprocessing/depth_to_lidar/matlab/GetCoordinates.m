% Convert spherical coordinates into cartesian with correction
function [x,y,z] = GetCoordinates(values,angles)
    nPoints = length(values);
    c = zeros(nPoints,3);
    
   for i = 1:nPoints
       v = degtorad(angles(i,1));
       h = degtorad(angles(i,2));
       r = values(i);

       [x,y,z] = sph2cart(h,v,r);

       % Depth is the perpendicular distance between a point and the image
       % plane, so we need some correction for this.
       c(i,1) = (1/cos(h))*(1/cos(v))*x;
       c(i,2) = (1/cos(h))*(1/cos(v))*y;
       c(i,3) = (1/cos(h))*(1/cos(v))*z;
   end

    x = c(:,1);
    y = c(:,2);
    z = c(:,3);
end

