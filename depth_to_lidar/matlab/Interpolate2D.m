function interpolatedValues = Interpolate2D(V,Xquery,Yquery, threshold)

    % Four pixels with indices cX and cY:
    % (cX1,cY1) (cX2,cY1)
    % (cX1,cY2) (cX2,cY2)
    
    nPoints = length(Xquery);
    interpolatedValues = zeros(nPoints,1);
    
    % For each point
    for iPoint = 1:nPoints
        
        % Instantiate coordinate variables
        cX1 = 0; cX2 = 0; cY1 = 0; cY2 = 0;
        fx = 0; fy = 0;
        
        % Get the query point coordinates
        Xq = Xquery(iPoint);
        Yq = Yquery(iPoint);
        
        % Get horizontal pixel index on which the point lies
        cX = floor(Xq);

        % decimal part of point from pixel's beginning
        decimalX = Xq - cX;
        
        % If the queried point is to the left of the pixel's center
        if decimalX <= 0.5
            cX1 = cX; % Set cX1 to the be the pixel index of cX
            cX2 = cX - 1; % We're interested in the pixel to the left of cX
            fx = decimalX + 0.5;
        else
            cX1 = cX + 1; % We're interested in the pixel to the right of cX
            cX2 = cX; % Set cX2 to the be the pixel index of cX
            fx = decimalX - 0.5;
        end
        
        % Repeat procedure for y-axis
        cY = floor(Yq);
        decimalY = Yq - cY;

        if decimalY <= 0.5
            cY1 = cY;
            cY2 = cY - 1;
            fy = decimalY + 0.5;
        else
            cY1 = cY + 1;
            cY2 = cY;
            fy = decimalY - 0.5;
        end
        
        % Get value in each pixel of relevance
        v1 = V(cY1,cX1);
        v2 = V(cY1,cX2);
        v3 = V(cY2,cX1);
        v4 = V(cY2,cX2);
        
        % Calculate difference in pixel intensity between pixel hit by
        % query point and the other three pixels.
       	v = V(cY,cX);
        d1 = abs(v - v1);
        d2 = abs(v - v2);
        d3 = abs(v - v3);
        d4 = abs(v - v4);
        
        c = 0;
        %If difference is larger than threshold, do not interpolate.
        if d1 > threshold || d2 > threshold || d3 > threshold || d4 > threshold 
            c = v;
        else
            % Linearly interpolate between pixel values to find value in query
            % point.
            a = v1*fx + v2*(1-fx);
            b = v3*fx + v4*(1-fx);
            c = a*fy + b*(1-fy);
        end
        
        interpolatedValues(iPoint) = c;
    end
end

