function [translation_frame,projerr_no_rot] = global_vals2(frame_keypts3D,frame_keypts2D)
f = 1000; % Focal lenth of camera
vidpos3D = frame_keypts3D;
keypts2D = frame_keypts2D;

m = 10;
pxy = cell(1,m);
kxy = cell(1,m);

% % 1) Pelvis
pxy{1} = [vidpos3D(1,1);vidpos3D(1,2)];
% take pelvis as mean of left and right hip
kxy{1} = [(keypts2D(3)+keypts2D(5))/2;(keypts2D(4)+keypts2D(6))/2];

% % 1) right hip
% pxy{1} = [vidpos3D(2,1);vidpos3D(2,2)];
% % take pelvis as mean of left and right hip
% kxy{1} = [keypts2D(5);keypts2D(6)];

% 2) Left Knee
pxy{2} = [vidpos3D(6,1);vidpos3D(6,2)];
kxy{2} = [keypts2D(15);keypts2D(16)];

% 3) Left Ankle
pxy{3} = [vidpos3D(7,1);vidpos3D(7,2)];
kxy{3} = [keypts2D(19);keypts2D(20)];

% 4) Right Knee
pxy{4} = [vidpos3D(3,1);vidpos3D(3,2)];
kxy{4} = [keypts2D(17);keypts2D(18)];

% 5) Right Ankle
pxy{5} = [vidpos3D(4,1);vidpos3D(4,2)];
kxy{5} = [keypts2D(21);keypts2D(22)];

% 6) Nose
% try nose as mid neck
pxy{6} = [vidpos3D(10,1);vidpos3D(10,2)];
kxy{6} = [keypts2D(1);keypts2D(2)];

% 7) Right Shoulder
pxy{7} = [vidpos3D(15,1);vidpos3D(15,2)];
kxy{7} = [keypts2D(13);keypts2D(14)];

% 8) Right Wrist
pxy{8} = [vidpos3D(17,1);vidpos3D(17,2)];
kxy{8} = [keypts2D(9);keypts2D(10)];

% 9) Left Shoulder
pxy{9} = [vidpos3D(12,1);vidpos3D(12,2)];
kxy{9} = [keypts2D(11);keypts2D(12)];

% 10) Left Wrist
pxy{10} = [vidpos3D(14,1);vidpos3D(14,2)];
kxy{10} = [keypts2D(7);keypts2D(8)];

% % 11) left hip
% pxy{11} = [vidpos3D(5,1);vidpos3D(5,2)];
% % take pelvis as mean of left and right hip
% kxy{11} = [keypts2D(3);keypts2D(4)];

% for i = 1:10
% disp(['Pxy = ', num2str(pxy{i}(1)),',', num2str(pxy{i}(2))])
% disp(['Kxy = ', num2str(kxy{i}(1)),',', num2str(kxy{i}(2))])
% disp(i)
% disp('-----')
% end

tot_pxy = pxy{1};
tot_kxy = kxy{1};
for i = 2:m
    tot_pxy = pxy{i} + tot_pxy;
    tot_kxy = kxy{i} + tot_kxy;
end
mean_pxy = tot_pxy/m;
mean_kxy = tot_kxy/m;

squaremagpxy_sum = 0;
squaremagkxy_sum = 0;
for i = 1:m
    % numerator
    pxnormalized = pxy{i}(1)-mean_pxy(1);
    pynormalized = pxy{i}(2)-mean_pxy(2);
    squaremagpxy_temp = pxnormalized^2 + pynormalized^2;
    squaremagpxy_sum = squaremagpxy_sum + squaremagpxy_temp;    
    
    % denominator
%     kxnormalized = kxy{i}(1)-mean_kxy(1);
%     kynormalized = kxy{i}(2)-mean_kxy(2);
%     squaremagkxy_temp = kxnormalized*pxnormalized + kynormalized*pynormalized
%     squaremagkxy_sum = squaremagkxy_sum + squaremagkxy_temp; 

    kxnormalized = kxy{i}(1)-mean_kxy(1);
    kynormalized = kxy{i}(2)-mean_kxy(2);
    squaremagkxy_temp = kxnormalized^2 + kynormalized^2;
    squaremagkxy_sum = squaremagkxy_sum + squaremagkxy_temp;    
    
end
% numerator = squaremagpxy_sum
% denominator = squaremagkxy_sum

numerator = sqrt(squaremagpxy_sum);
denominator = sqrt(squaremagkxy_sum);

z = f*numerator/denominator;
y = mean_kxy(2)*z/f-mean_pxy(2);
x = mean_kxy(1)*z/f-mean_pxy(1);
translation_frame = [x,y,z];

pimat = [f/z,0,0;0,f/z,0];

% rotated_vals = rotate_fov(imagedims(2),bbox,z);
% 
% projerr_rot = kxy{i}-pimat*rotated_vals*(([x;y;z]+[vidpos3D(j,1);vidpos3D(j,2);vidpos3D(j,3)]))
i = 10
j = 14
ind = 1
% Projection error for a sample of the joints for rotated and non-rotated
% coordinates to show that rotated coordinates have a lower error rate due
% to it corresponding to camera FOV instead of bounding box FOV (If the
% difference if FOV angle is more than 3 degrees).
projerr_no_rot_temp(ind,:) = kxy{i}-pimat*([x;y;z]+[vidpos3D(j,1);vidpos3D(j,2);vidpos3D(j,3)]);

i = 1;
j = 1;
ind = 2;
projerr_no_rot_temp(ind,:) = kxy{i}-pimat*([x;y;z]+[vidpos3D(j,1);vidpos3D(j,2);vidpos3D(j,3)]);
i = 5;
j = 4;
ind = 3;


projerr_no_rot_temp(ind,:) = kxy{i}-pimat*(([x;y;z]+[vidpos3D(j,1);vidpos3D(j,2);vidpos3D(j,3)]));

i = 6;
j = 10;
ind = 4;

projerr_no_rot_temp(ind,:) = kxy{i}-pimat*(([x;y;z]+[vidpos3D(j,1);vidpos3D(j,2);vidpos3D(j,3)]));


projerr_no_rot(1) = sum(projerr_no_rot_temp(:,1));
projerr_no_rot(2) = sum(projerr_no_rot_temp(:,2));