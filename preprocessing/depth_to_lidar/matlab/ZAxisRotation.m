function [X,Y,Z] = ZAxisRotation(x,y,z,theta)
    X = x*cos(theta) + y*sin(theta);
    Y = -x*sin(theta) + y*cos(theta);
    Z = z;
end

