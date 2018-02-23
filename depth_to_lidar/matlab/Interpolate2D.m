function interpolatedValues = Interpolate2D(V, Xquery, Yquery, threshold)

    % Four pixels with indices cX and cY:
    % (x1,y1) (x2,y1)
    % (x1,y2) (x2,y2)
    
    nPoints = length(Xquery);
    interpolatedValues = zeros(nPoints,1);
    height = size(V,1);
    width = size(V,2);
    
    % For each point
    for iPoint = 1:nPoints
        
        % Get the query point coordinates
        x = Xquery(iPoint);
        y = Yquery(iPoint);

        boundL = x < 0.5;               % Are we at left boundary?
        boundR = x > size(V,2) - 0.5;   % Are we at right boundary?
        boundT = y < 0.5;               % Are we at upper boundary?
        boundB = y > size(V,1) - 0.5;   % Are we att lower boundary?
        
        % Get the pixel index which the query point lies in.
        % Here we assume pixel indices start at 0.
        if boundR
            xp = width - 1;
        else
            xp = floor(x);
        end
        
        if boundB
            yp = height - 1;
        else
            yp = floor(y);
        end
        
        decimalX = x - xp;
        decimalY = y - yp;
        
        if decimalX < 0.5
            x1 = xp - 1;   % Set cX1 to the be the pixel to the left of cX
            x2 = xp;       % The query point is in pixel cX2
        else
            x1 = xp;       % We're interested in the pixel to the right of cX
            x2 = xp + 1;   % Set cX2 to the be the pixel index of cX
        end
        
        if decimalY < 0.5
            y1 = yp - 1;   
            y2 = yp;
        else
            y1 = yp;       
            y2 = yp + 1;
        end
        

        % The most common case is to not be on any boundary
        if ~(boundL || boundR || boundT || boundB)
            
            % Get value in each pixel of relevance
            v1 = V(y1 + 1,x1 + 1);
            v2 = V(y1 + 1,x2 + 1);
            v3 = V(y2 + 1,x1 + 1);
            v4 = V(y2 + 1,x2 + 1);
            
            % Interpolate only if difference between any pair of depth
            % values does not exceed threshold.
            if max([v1,v2,v3,v4]) - min([v1,v2,v3,v4]) < threshold
                a = v3*((x2+0.5) - x) + v4*(x - (x1+0.5));
                b = v1*((x2+0.5) - x) + v2*(x - (x1+0.5));
                c = b*((y2+0.5) - y) + a*(y - (y1+0.5));
            else
                c = V(yp + 1,xp + 1);
            end
    
        % If we are in a corner pixel
        elseif (boundL || boundR) && (boundT || boundB)
            
            % Just return pixel value without interpolating
            c = V(yp + 1,xp + 1);
         
        elseif boundT % If we are at top boundary
            
            % Interpolate in x only
            v3 = V(y2 + 1,x1 + 1);
            v4 = V(y2 + 1,x2 + 1);
            
            if max(v3,v4) - min(v3,v4) < threshold
                c = v3*((x2+0.5) - x) + v4*(x - (x1+0.5));
            else
                c = V(yp + 1,xp + 1);
            end
            
        elseif boundB % If we are at bottom boundary
            
            % Interpolate in x only
            v1 = V(y1 + 1,x1 + 1);
            v2 = V(y1 + 1,x2 + 1);
            
            if max(v1,v2) - min(v1,v2) < threshold
                c = v1*((x2+0.5) - x) + v2*(x - (x1+0.5));
            else
                c = V(yp + 1,xp + 1);
            end

        elseif boundL % If we are at left boundary
            
            % Interpolate in y only
            v2 = V(y1 + 1,x2 + 1);
            v4 = V(y2 + 1,x2 + 1);
            
            if max(v2,v4) - min(v2,v4) < threshold
                c = v2*((y2+0.5) - y) + v4*(y - (y1+0.5));
            else
                c = V(yp + 1,xp + 1);
            end
                
        else % We must be at right boundary
            
            % Interpolate in y only
            v1 = V(y1 + 1,x1 + 1);
            v3 = V(y2 + 1,x1 + 1);
            
            if max(v2,v4) - min(v2,v4) < threshold
                c = v1*((y2+0.5) - y) + v3*(y - (y1+0.5));
            else
                c = V(yp + 1,xp + 1);
            end
        end
        
        interpolatedValues(iPoint) = c;

    end
end

