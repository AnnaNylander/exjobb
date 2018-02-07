function depthImage = DecodeDepth(imageData, farPlaneDistance)
    imageData = single(imageData);
    depthImage = imageData(:,:,1) + 256.*imageData(:,:,2) + (256*256).*imageData(:,:,3);
    depthImage = depthImage / (256*256*256-1);
    depthImage = depthImage * farPlaneDistance;
end

