function rotationMatrix = Roty(theta)
    theta = degtorad(theta);
    rotationMatrix = [cos(theta), 0         , sin(theta)          ;
                      0, 1, 0;
                      -sin(theta), 0, cos(theta)];
end