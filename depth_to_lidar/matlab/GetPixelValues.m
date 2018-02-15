function pixelValues = GetPixelValues(imageA, xCoords, yCoords)
    xCoords = int32(xCoords);
    yCoords = int32(yCoords);
    
    nPoints = length(xCoords);
    pixelValues = zeros(nPoints, 1);
    
    for iPoint = 1:nPoints
        pixelValues(iPoint) = imageA(yCoords(iPoint), xCoords(iPoint));
    end
    
end

