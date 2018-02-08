function [x,y,z] = GetCoordinates(image,anglesV,anglesH, distanceV, distanceH)
    nHorizontal = length(distanceH);
    nVertical = length(distanceV);
    c = zeros(nVertical*nHorizontal,3);
    
    for i = 1:nVertical
       for j = 1:nHorizontal
           v = anglesV(i);
           h = anglesH(j);
           r = image(distanceV(i),distanceH(j));

           pointIndex = (i-1)*nHorizontal + j;
            
           [x,y,z] = sph2cart(h,v,r);
           %c(pointIndex,1) = r * cos(h) * cos(v);
           %c(pointIndex,2) = r * sin(h) * cos(v);
           %c(pointIndex,3) = r * sin(v);
           
           c(pointIndex,1) = (1/cos(h))*x;
           c(pointIndex,2) = (1/cos(h))*y;
           c(pointIndex,3) = z;
           
           %c(pointIndex,1) = x;
           %c(pointIndex,2) = y;
           %c(pointIndex,3) = z;
       end
    end

    x = c(:,1);
    y = c(:,2);
    z = c(:,3);
end

