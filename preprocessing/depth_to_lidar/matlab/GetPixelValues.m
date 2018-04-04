function pixelValues = GetPixelValues(imageA, xCoords, yCoords)
    % For x-coordinates, we want to quantize to the left
    xCoords = floor(xCoords) + 1;
    % For y-coordinates, we want to quantize downwards
    yCoords = ceil(yCoords);
    
    nPoints = length(xCoords);
    pixelValues = zeros(nPoints, 1);
    
    for iPoint = 1:nPoints
        pixelValues(iPoint) = imageA(yCoords(iPoint) , xCoords(iPoint));
    end
    
end

