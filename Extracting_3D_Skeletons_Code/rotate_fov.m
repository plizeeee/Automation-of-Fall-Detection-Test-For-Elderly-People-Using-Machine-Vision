function rotated_vals = rotate_fov(imagexdim,bbox,dz,f)
    f = 1000;
    im_center_x = imagexdim/2;
    bboxcenter_x = (bbox(3)+bbox(1))/2;
    % convert pixel coordinates to global coordinates with weak perspective
    % projection
    im_center_x_globalcoords = dz/f*im_center_x;
    bboxcenter_x_globalcoords = dz/f*bboxcenter_x;
    
    dx = im_center_x_globalcoords-bboxcenter_x_globalcoords;
    if dz>0
        theta = atan(dx/dz);
    else
        theta = atan(dx/dz)+pi;
    end
        rotated_vals= [cos(theta),0,sin(theta);0,1,0;-sin(theta),0,cos(theta)]; %Rotation matrix
end