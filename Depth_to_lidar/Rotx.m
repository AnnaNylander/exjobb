function rotationMatrix = Rotz(theta)
    theta = degtorad(theta);
    rotationMatrix = [1, 0         , 0          ;
                      0, cos(theta), -sin(theta);
                      0, sin(theta), cos(theta)];
end