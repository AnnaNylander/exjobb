function [x,y,z] = TrimToRoi(xyz, roi)
    xyz = xyz(max(abs(xyz),[],2) < roi/2, :);
    x = xyz(:,1);
    y = xyz(:,2);
    z = xyz(:,3);
end

