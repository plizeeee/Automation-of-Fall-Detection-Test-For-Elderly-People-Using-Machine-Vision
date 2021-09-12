clear
clc
close all
% trial_number = 2
frame_rate = 28.3
% frame_rate = 28
% framenum = round(frame_rate*18.3) % Plot pose of this frame
framenum = 100
%%%%%

for subject_id = 1:30
   for trial_number = [1,2,4,5]
       %%%%%%%%%%%%%%%%%%%%%%%%%%
       
        keypts2D = csvread(['output/subjects/subject',num2str(subject_id), '/trial', num2str(trial_number),'/2dpose.csv'])
        bboxes = csvread(['output/subjects/subject',num2str(subject_id), '/trial', num2str(trial_number),'/bboxes.csv'])
        keypts_1Darr = csvread(['output/subjects/subject',num2str(subject_id), '/trial', num2str(trial_number),'/rgb_skeleton.csv'])
        keypts_1Darr_stage6_vid = csvread(['output/subjects/subject',num2str(subject_id), '/trial', num2str(trial_number),'/rgb_skeleton.csv'])

        predictions = cell(1,1);
        pose3D = cell(1,size(keypts_1Darr,1));
        for i = 1:6
            pose3D{i} = reshape(keypts_1Darr(i,:), [17,3]);
        end


        % pose for all frames
        n = size(keypts_1Darr_stage6_vid,1)
        vidpos3D = cell(1,n)
        for i = 1:n
            vidpos3D{i} = reshape(keypts_1Darr_stage6_vid(i,:), [17,3]);
        end

        skel_vid_frame = vidpos3D{framenum};
        % 
        %  figure;
        %     % image used for testing
        %     imshow(img);
        %     title('Test Image');
        %     figure;
        %     % Estimated 3D poses corresponding to each stage (1 - 6)
        %     set(gcf, 'Position', get(0,'Screensize'));
        %     for i = 1 : 6
        %         subplot(2, 3, i);
        %         plotSkel3D(pose3D{i}, 'r');
        %         title(sprintf('Pose 3D prediction - stage %d', i));
        %     end
        %     

        frame_diff = 0;
        ind1 = 280;
        ind2 = ind1+frame_diff;

        % Computing global position of person
        reference_frame = cell(1,n);
        rot_mat = cell(1,n);
        frame_keypts2D = cell(1,n);
        frame_keypts3D = cell(1,n);
        projerror_rot = zeros(2,n);
        skel_vid_frame_new = zeros(17,3);
        vid_global_skeletons = cell(1,n)

        im_dims = [1080,1920,3];
        ind1 = 1;
        ind2 = ind1+frame_diff;
        frame_keypts3D{ind1} = vidpos3D{ind1};
        frame_keypts2D{ind1} = keypts2D(:,ind2);
        [reference_frame{ind1},projerror_rot(:,ind1)] = global_pose_from_relative_pose(frame_keypts3D{ind1},frame_keypts2D{ind1});
        z = reference_frame{ind1}(3);

        for ind1 = 2:n
            ind2 = ind1+frame_diff;
            rotated_vals = rotate_fov(im_dims(2),bboxes(:,ind1),z);
            frame_keypts3D{ind1} = vidpos3D{ind1};
            for kk = 1:17
               % Rotated frame to allign with FOV of camera
               frame_keypts3D{ind1}(kk,:) =  (rotated_vals*frame_keypts3D{ind1}(kk,:)')';
            end
            frame_keypts2D{ind1} = keypts2D(:,ind2);

            [reference_frame{ind1},projerror_rot(:,ind1)] = global_pose_from_relative_pose(frame_keypts3D{ind1},frame_keypts2D{ind1});
            z = reference_frame{ind1}(3);
            for i = 1:17
                skel_vid_frame_new(i,:) = (frame_keypts3D{ind1}(i,:)+reference_frame{ind1});
            end
            vid_global_skeletons{ind1} = skel_vid_frame_new;
            if ind1 ==2 % Duplicate first frame to keep signal length consistent for processing
                vid_global_skeletons{1} = skel_vid_frame_new;
            end
        end

        % Projection error to show goodness of fit
        mean_proj_error_rot(1) = mean(abs(projerror_rot(1,:)));
        mean_proj_error_rot(2) = mean(abs(projerror_rot(2,:)));
        disp('results:')
        disp(mean_proj_error_rot)
        % Save global skeleton to csv file
        global_skeleton_unfolded = zeros(size(vid_global_skeletons,2),51);
        for i = 1:size(vid_global_skeletons,2)
            global_skeleton_unfolded(i,:) = reshape(vid_global_skeletons{i},[51,1]);
        end
            csvwrite(['output/subjects/subject',num2str(subject_id), '/trial', num2str(trial_number),'/global_rgb_skeleton.csv'],global_skeleton_unfolded)
   end
end