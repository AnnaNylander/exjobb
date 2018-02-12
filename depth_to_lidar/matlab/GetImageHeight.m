% Calculate the required image height to match a given vertical FOV.
function imgHeight = GetImageHeight(imgWidth,fovH,fovV,normal,p)
     d = imgWidth/(2*tan(fovH/2));
     p3 = [d,0,0];
     lineP = p3*Roty(-fovV/2); % rotate down
     lineP = lineP*Rotz(-fovH/2); % rotate left
     [pixel, check] = plane_line_intersect(normal,p,lineP,[0 0 0]);
     imgHeight = abs(2*pixel(3));
end

