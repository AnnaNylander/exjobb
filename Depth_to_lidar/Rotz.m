function rotationMatrix = Rotz(theta)
    theta = degtorad(theta);
    rotationMatrix = [cos(theta),-sin(theta),0;
                      sin(theta),cos(theta), 0;
                      0         ,0         , 1];
end

