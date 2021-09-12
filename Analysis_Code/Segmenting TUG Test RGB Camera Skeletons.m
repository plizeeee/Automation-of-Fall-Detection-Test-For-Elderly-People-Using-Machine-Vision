clear
clc

type_trial = 1; % 1 = trials 1,2 (1.5m)  and 4 = trials 4,5 (3m)

% After reducing the size of the script, avg time ~ 8 seconds per parameter

frame_rate = 28.3

% Search parameter space
% The space is explored seperately for each part of the TUG test, since they are independant of each other and performing the search independandly reduces the search time by several orders of magnitude

% Start test
param_starttest_drop = [1.05,1.09,1.12,1.17,1.22,1.3,1.4,1.55,1.7,2,2.5,3,3.5,4,5,6,7,8];
param_starttest_window = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
param_starttest_epsilon = [0,0.5,1,2,3,4,6,8,10,12,15,18,21,24,27,30,35,40,45,50];

% Stand
param_stand_drop = [1.05,1.09,1.12,1.17,1.22,1.3,1.4,1.55,1.7,2,2.5,3,3.5,4,5,6,7,8];
param_stand_window = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
param_stand_epsilon = [0,0.5,1,2,3,4,6,8,10,12,15,18,21,24,27,30,35,40,45,50];

% Start Turn
param_startturn_drop = [1.05,1.09,1.12,1.17,1.22,1.3,1.4,1.55,1.7,2,2.5,3,3.5,4,5,6,7,8];
param_startturn_window = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
param_startturn_epsilon = [0,0.5,1,2,3,4,6,8,10,12,15,18,21,24,27,30,35,40,45,50];

% End Turn
param_midturn_drop = [1.05,1.09,1.12,1.17,1.22,1.3,1.4,1.55,1.7,2,2.5,3,3.5,4,5,6,7,8];
param_midturn_window = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
param_midturn_epsilon = [0,0.5,1,2,3,4,6,8,10,12,15,18,21,24,27,30,35,40,45,50];
param_endturn_drop = [1.05,1.09,1.12,1.17,1.22,1.3,1.4,1.55,1.7,2,2.5,3,3.5,4,5,6,7,8];
param_endturn_window = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
param_endturn_epsilon = [0,0.5,1,2,3,4,6,8,10,12,15,18,21,24,27,30,35,40,45,50];

% Start Sit
param_startsit_drop = [1.05,1.09,1.12,1.17,1.22,1.3,1.4,1.55,1.7,2,2.5,3,3.5,4,5,6,7,8];
param_startsit_window = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
param_startsit_epsilon = [0,0.5,1,2,3,4,6,8,10,12,15,18,21,24,27,30,35,40,45,50];

% End Sit
param_midsit_drop = [1.05,1.09,1.12,1.17,1.22,1.3,1.4,1.55,1.7,2,2.5,3,3.5,4,5,6,7,8];
param_midsit_window = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
param_midsit_epsilon = [0,0.5,1,2,3,4,6,8,10,12,15,18,21,24,27,30,35,40,45,50];
param_endsit_drop = [1.05,1.09,1.12,1.17,1.22,1.3,1.4,1.55,1.7,2,2.5,3,3.5,4,5,6,7,8];
param_endsit_window = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
param_endsit_epsilon = [0,0.5,1,2,3,4,6,8,10,12,15,18,21,24,27,30,35,40,45,50];

tic % Timing code

% Random Permutations for Cross Validation
rand_order = randperm(60)+9;
training_subjects = rand_order(1:48)
test_subjects = rand_order(49:60)

cohort = cell(5,1)
cohort{1} = rand_order(1:12)
cohort{2} = rand_order(13:24)
cohort{3} = rand_order(25:36)
cohort{4} = rand_order(37:48)
cohort{5} = rand_order(49:60)

program_state = 1
      % If program crashes, load most recent state
% load('cohort_most_recent.mat')


%%%%%%%%%%%%%% Pre loading values for faster computation
zplotvals_pelvis_global_smooth = cell(30,2);
yplotvals_height_smooth = cell(30,2);
xplotvals_left_hip_smooth = cell(30,2);
xplotvals_right_hip_smooth = cell(30,2);
xplotvals_hip_absdiff_smooth =  cell(30,2);
zplotvals_right_foot_smooth = cell(30,2);
zdiff_smooth = cell(30,2);
zplotvals_pos_shoulders_smooth = cell(30,2);

zplotvals_vel_shoulders = cell(30,2);
zplotvals_vel_shoulders_smooth = cell(30,2);

yplotvals_height_vel_shoulders = cell(30,2);
yplotvals_height_vel_shoulders_smooth = cell(30,2);

zplotvals_acc_shoulders = cell(30,2);
zplotvals_acc_shoulders_smooth = cell(30,2);
times_groundtruth = cell(30,2)

% Preloading skeleton values
for subject_id = 1:30
    for trial_number = type_trial : type_trial + 1
		vid_global_skels_unfolded = csvread(['output/subjects/subject',num2str(subject_id), '/trial', num2str(trial_number),'/global_rgb_skeleton.csv']); 
		vid_local_skels = csvread(['output/subjects/subject',num2str(subject_id), '/trial', num2str(trial_number),'/rgb_skeleton.csv']);
		times_groundtruth{subject_id,trial_number-type_trial + 1} = csvread(['output/subjects/subject',num2str(subject_id), '/trial', num2str(trial_number),'/segmented_times_mean.csv']);

		% Fold skeleton values into useful format
		n = size(vid_global_skels_unfolded,1);
		vid_global_skeletons = cell(1,n);
		vidpos3D = cell(1,n)
		for i = 1:n
			vid_global_skeletons{i} = reshape(vid_global_skels_unfolded(i,:),[17,3]);
			vidpos3D{i} = reshape(vid_local_skels(i,:), [17,3]);
		end
		yplotvals_height = zeros(1,n);
		xplotvals_left_hip = zeros(1,n);
		xplotvals_right_hip = zeros(1,n);
		zplotvals_feet_diff = zeros(1,n);
		yplotvals_hip_knee_absdiff = zeros(1,n);
		zplotvals_pelvis_global = zeros(1,n);
		xplotvals_hip_absdiff2 = zeros(1,n);
		xplotvals_hip_absdiff = zeros(1,n);
		zplotvals_right_foot = zeros(1,n);
		zdiff = zeros(1,n);
		zplotvals_pos_shoulders = zeros(1,n);

		for i = 1:n
			% Relative positions
%             yplotvals_height(i) = -vidpos3D{i}(11,2);
			yplotvals_height(i) = -(vidpos3D{i}(11,2)-(vidpos3D{i}(4,2)+vidpos3D{i}(7,2))/2);
%             yplotvals_height(i) = vidpos3D{i}(4,2);
			xplotvals_left_hip(i) = vidpos3D{i}(13,1);
			xplotvals_right_hip(i) = vidpos3D{i}(16,1);
			zplotvals_feet_diff(i) = (vidpos3D{i}(4,3)-vidpos3D{i}(7,3))/2;
			zplotvals_pos_shoulders(i) = (vid_global_skeletons{i}(12,3)+vid_global_skeletons{i}(15,3))/2;


		%     yplotvals_pelvis(i) = vid_global_skeletons{i}(1,2)

			% Checking if hips are parralel to knees for when they sit back down
			yplotvals_hip_knee_absdiff(i) = abs(vidpos3D{i}(2,2)-vidpos3D{i}(3,2))+abs(vidpos3D{i}(5,2)-vidpos3D{i}(6,2));
			% The z value is actually the depth of the image, it's  just represented differently for the pinhole camera model 
			% Hip gloval y pos over time
			zplotvals_pelvis_global(i) = vid_global_skeletons{i}(1,3);
			xplotvals_hip_absdiff2(i) = abs(vid_global_skeletons{i}(16,1)-vid_global_skeletons{i}(13,1));
			xplotvals_hip_absdiff(i) = abs(vidpos3D{i}(5,1)-vidpos3D{i}(2,1));
			zplotvals_right_foot(i) = vid_global_skeletons{i}(7,3) + vid_global_skeletons{i}(4,3);
			zdiff(i) = (vid_global_skeletons{i}(7,3) + vid_global_skeletons{i}(4,3))/2-(vid_global_skeletons{i}(11,3)+vid_global_skeletons{i}(12,3)+vid_global_skeletons{i}(15,3))/3;
		end

		zplotvals_pelvis_global_smooth{subject_id,trial_number-type_trial+1} = smoothdata(zplotvals_pelvis_global,'movmedian',8);
		yplotvals_height_smooth{subject_id,trial_number-type_trial+1} = smoothdata(yplotvals_height,'movmedian',8);
		xplotvals_left_hip_smooth{subject_id,trial_number-type_trial+1} = smoothdata(xplotvals_left_hip,'movmedian',8);
		xplotvals_right_hip_smooth{subject_id,trial_number-type_trial+1} = smoothdata(xplotvals_right_hip,'movmedian',8);
		xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1} =  smoothdata(xplotvals_hip_absdiff,'movmedian',8);
		zplotvals_right_foot_smooth{subject_id,trial_number-type_trial+1} = smoothdata(zplotvals_right_foot,'movmedian',8);
		zdiff_smooth{subject_id,trial_number-type_trial+1} = smoothdata(zdiff,'movmedian',8);
		zplotvals_pos_shoulders_smooth{subject_id,trial_number-type_trial+1} = smoothdata(zplotvals_pos_shoulders,'movmedian',17);

		zplotvals_vel_shoulders{subject_id,trial_number-type_trial+1} = diff(zplotvals_pos_shoulders_smooth{subject_id,trial_number-type_trial+1});
		zplotvals_vel_shoulders_smooth{subject_id,trial_number-type_trial+1} = smoothdata(zplotvals_vel_shoulders{subject_id,trial_number-type_trial+1},'movmedian',3);

		yplotvals_height_vel_shoulders{subject_id,trial_number-type_trial+1} = diff(yplotvals_height_smooth{subject_id,trial_number-type_trial+1});
		yplotvals_height_vel_shoulders_smooth{subject_id,trial_number-type_trial+1} = smoothdata(yplotvals_height_vel_shoulders{subject_id,trial_number-type_trial+1},'movmedian',3);

		zplotvals_acc_shoulders{subject_id,trial_number-type_trial+1} = diff(zplotvals_vel_shoulders_smooth{subject_id,trial_number-type_trial+1});
		zplotvals_acc_shoulders_smooth{subject_id,trial_number-type_trial+1} = smoothdata(zplotvals_acc_shoulders{subject_id,trial_number-type_trial+1},'movmedian',2);
    end
end

tic
for cohort_ind = program_state:5
    training_subjects = []
    test_subjects = []
    for include = 1:5
        if include~=cohort_ind
            training_subjects = [training_subjects,cohort{include}]
        else
            test_subjects = cohort{include}
        end
    end
    
    param_ind = 0;
    error_starttest_params = []
    error_stand_params = []
    error_startturn_params = []
    error_endturn_params = []
    error_startsit_params = []
    error_endsit_params = []

    for epsilon_ind = 1:20
        for window_ind = 1:14
            for drop_ind= 1:18
                param_ind = param_ind + 1;
                z = 1;
                zk = 1;
                for chosen_subject_trails_pairs = training_subjects
                        
                    subject_id = floor(chosen_subject_trails_pairs/2);
                    trial_number = mod(chosen_subject_trails_pairs,2)+type_trial; % + 1 for trials 1 and 2
                    m = 1; % Start frame (start will normally just be the first frame of the video, since user will be seated

                    tot_drop = floor((max(yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(m:end))-(yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(1)))/param_stand_drop(drop_ind));
                    stand_up_ind_rgb  = after_drop(-yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(m:end), tot_drop, param_stand_window(window_ind),param_stand_epsilon(epsilon_ind)) +m-1;

					flipped_zplotvals_pos_shoulders_smooth = flip(zplotvals_pos_shoulders_smooth{subject_id,trial_number-type_trial+1}(m:times_groundtruth{subject_id,trial_number-type_trial + 1}(2)));
					tot_drop = (max(flipped_zplotvals_pos_shoulders_smooth(m:times_groundtruth{subject_id,trial_number-type_trial + 1}(2)))-min(flipped_zplotvals_pos_shoulders_smooth(m:times_groundtruth{subject_id,trial_number-type_trial + 1}(2))))/param_starttest_drop(drop_ind);        
					start_test  = before_drop(zplotvals_pos_shoulders_smooth{subject_id,trial_number-type_trial+1}(m:times_groundtruth{subject_id,trial_number-type_trial + 1}(2)), tot_drop, param_starttest_window(window_ind),param_starttest_epsilon(epsilon_ind)) +m-1;

					tot_drop = floor((max(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(times_groundtruth{subject_id,trial_number-type_trial + 1}(2):end))-min(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(times_groundtruth{subject_id,trial_number-type_trial + 1}(2):end)))/param_startturn_drop(drop_ind));
					reach_line_ind_rgb = before_drop(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(times_groundtruth{subject_id,trial_number-type_trial + 1}(2):end),tot_drop,param_startturn_window(window_ind),param_startturn_epsilon(epsilon_ind))+times_groundtruth{subject_id,trial_number-type_trial + 1}(2)-1;

					tot_drop = floor((max(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1})-min(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}))/param_midturn_drop(drop_ind));
					mid_turn = after_drop(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}, tot_drop, param_midturn_window(window_ind),param_midturn_epsilon(epsilon_ind));

					tot_drop = floor((max(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(mid_turn:end))-min(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(mid_turn:end)))/param_endturn_drop(drop_ind));
					turn_complete_ind_rgb =  after_drop(-xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(mid_turn:end), tot_drop, param_endturn_window(window_ind),param_endturn_epsilon(epsilon_ind))+mid_turn(1)-1;

					tot_drop = floor(((max(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(times_groundtruth{subject_id,trial_number-type_trial + 1}(4)-1:end-50))-(min(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(times_groundtruth{subject_id,trial_number-type_trial + 1}(4)-1:end-50))))/param_startsit_drop(drop_ind)));

					rdy_tosit_rgb = before_drop(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(times_groundtruth{subject_id,trial_number-type_trial + 1}(4)-1:end),tot_drop,param_startsit_window(window_ind),param_startsit_epsilon(epsilon_ind))+times_groundtruth{subject_id,trial_number-type_trial + 1}(4)-2;
	  
					sit_pot_inds = floor(size(yplotvals_height_smooth{subject_id,trial_number-type_trial+1},2)/2);
					tot_drop = floor(((max(yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(sit_pot_inds:end))-(min(yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(sit_pot_inds:end))))/param_midsit_drop(drop_ind)));
					bend_to_sit_rgb = after_drop(yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(sit_pot_inds:end),tot_drop,param_midsit_window(window_ind),param_midsit_epsilon(epsilon_ind))+sit_pot_inds-1;
					tot_drop = floor(((max(yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(bend_to_sit_rgb:end))-(min(yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(bend_to_sit_rgb:end))))/param_endsit_drop(drop_ind)));
					sit_rgb = after_drop(-yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(bend_to_sit_rgb:end), tot_drop, param_endsit_window(window_ind),param_endsit_epsilon(epsilon_ind)) +bend_to_sit_rgb-1;


					diff_times = times_groundtruth{subject_id,trial_number-type_trial + 1}-[start_test,stand_up_ind_rgb,reach_line_ind_rgb,turn_complete_ind_rgb,rdy_tosit_rgb,sit_rgb];
					temp_diff1(z) = abs(diff_times(1));
					temp_diff2(z)  = abs(diff_times(2));
					temp_diff3(z)  = abs(diff_times(3));
					temp_diff4(z)  = abs(diff_times(4));
					temp_diff5(z)  = abs(diff_times(5));
					temp_diff6(z)  = abs(diff_times(6));

					temp_diff1_signed(z) = (diff_times(1));
					temp_diff2_signed(z)  = (diff_times(2));
					temp_diff3_signed(z)  = (diff_times(3));
					temp_diff4_signed(z)  = (diff_times(4));
					temp_diff5_signed(z)  = (diff_times(5));
					temp_diff6_signed(z)  = (diff_times(6));

					rgb_vec = [start_test,stand_up_ind_rgb,reach_line_ind_rgb,turn_complete_ind_rgb,rdy_tosit_rgb,sit_rgb];
					z = z+1;
                      
            end
                disp(['param index = ' , num2str(param_ind)])
                error_starttest_params(param_ind) = mean(temp_diff1)/frame_rate;
                error_stand_params(param_ind) = mean(temp_diff2)/frame_rate;
                error_startturn_params(param_ind) = mean(temp_diff3)/frame_rate;
                error_endturn_params(param_ind) = mean(temp_diff4)/frame_rate;
                error_startsit_params(param_ind) = mean(temp_diff5)/frame_rate;
                error_endsit_params(param_ind) = mean(temp_diff6)/frame_rate;          
            end 
        end
    end

    z = 1;
    zk = 1;
	
    err = min(error_starttest_params)+ min(error_stand_params) + min(error_startturn_params) +...
        min(error_endturn_params) + min(error_startsit_params) + min(error_endsit_params)
	
	% Start test
    min_ind_starttest = find(min(error_starttest_params)==error_starttest_params)
    ind_param_starttest_drop = mod(min_ind_starttest(1)-1,length(param_starttest_drop))+1;
    ind_param_starttest_window = mod(floor((min_ind_starttest(1)-1)/length(param_starttest_drop)),length(param_starttest_window))+1;
    ind_param_starttest_epsilon = mod(floor((min_ind_starttest(1)-1)/(length(param_starttest_drop)*length(param_starttest_window))),length(param_starttest_epsilon))+1;
    param_starttest_drop_best = param_starttest_drop(ind_param_starttest_drop)
    param_starttest_window_best = param_starttest_window(ind_param_starttest_window)
    param_starttest_epsilon_best = param_starttest_epsilon(ind_param_starttest_epsilon)
	
	% Stand
    min_ind_stand = find(min(error_stand_params)==error_stand_params)
    ind_param_stand_drop = mod(min_ind_stand(1)-1,length(param_stand_drop))+1;
    ind_param_stand_window = mod(floor((min_ind_stand(1)-1)/length(param_stand_drop)),length(param_stand_window))+1;
    ind_param_stand_epsilon = mod(floor((min_ind_stand(1)-1)/(length(param_stand_drop)*length(param_stand_window))),length(param_stand_epsilon))+1;
    param_stand_drop_best = param_stand_drop(ind_param_stand_drop)
    param_stand_window_best = param_stand_window(ind_param_stand_window)
    param_stand_epsilon_best = param_stand_epsilon(ind_param_stand_epsilon)

	% Start turn
    min_ind_startturn = find(min(error_startturn_params)==error_startturn_params)
    ind_param_startturn_drop = mod(min_ind_startturn(1)-1,length(param_startturn_drop))+1;
    ind_param_startturn_window = mod(floor((min_ind_startturn(1)-1)/length(param_startturn_drop)),length(param_startturn_window))+1;
    ind_param_startturn_epsilon = mod(floor((min_ind_startturn(1)-1)/(length(param_startturn_drop)*length(param_startturn_window))),length(param_startturn_epsilon))+1;
    param_startturn_drop_best = param_startturn_drop(ind_param_startturn_drop)
    param_startturn_window_best = param_startturn_window(ind_param_startturn_window)
    param_startturn_epsilon_best = param_startturn_epsilon(ind_param_startturn_epsilon)

    % End turn
    min_ind_midturn = find(min(error_endturn_params)==error_endturn_params)
    ind_param_midturn_drop = mod(min_ind_midturn(1)-1,length(param_midturn_drop))+1;
    ind_param_midturn_window = mod(floor((min_ind_midturn(1)-1)/length(param_midturn_drop)),length(param_midturn_window))+1;
    ind_param_midturn_epsilon = mod(floor((min_ind_midturn(1)-1)/(length(param_midturn_drop)*length(param_midturn_window))),length(param_midturn_epsilon))+1;
    param_midturn_drop_best = param_midturn_drop(ind_param_midturn_drop)
    param_midturn_window_best = param_midturn_window(ind_param_midturn_window)
    param_midturn_epsilon_best = param_midturn_epsilon(ind_param_midturn_epsilon)

    min_ind_endturn = find(min(error_endturn_params)==error_endturn_params)
    ind_param_endturn_drop = mod(min_ind_endturn(1)-1,length(param_endturn_drop))+1;
    ind_param_endturn_window = mod(floor((min_ind_endturn(1)-1)/length(param_endturn_drop)),length(param_endturn_window))+1;
    ind_param_endturn_epsilon = mod(floor((min_ind_endturn(1)-1)/(length(param_endturn_drop)*length(param_endturn_window))),length(param_endturn_epsilon))+1;
    param_endturn_drop_best = param_endturn_drop(ind_param_endturn_drop)
    param_endturn_window_best = param_endturn_window(ind_param_endturn_window)
    param_endturn_epsilon_best = param_endturn_epsilon(ind_param_endturn_epsilon)

    % Start sit
    min_ind_startsit = find(min(error_startsit_params)==error_startsit_params)
    ind_param_startsit_drop = mod(min_ind_startsit(1)-1,length(param_startsit_drop))+1;
    ind_param_startsit_window = mod(floor((min_ind_startsit(1)-1)/length(param_startsit_drop)),length(param_startsit_window))+1;
    ind_param_startsit_epsilon = mod(floor((min_ind_startsit(1)-1)/(length(param_startsit_drop)*length(param_startsit_window))),length(param_startsit_epsilon))+1;
    param_startsit_drop_best = param_startsit_drop(ind_param_startsit_drop)
    param_startsit_window_best = param_startsit_window(ind_param_startsit_window)
    param_startsit_epsilon_best = param_startsit_epsilon(ind_param_startsit_epsilon)

    % End sit
    min_ind_midsit = find(min(error_endsit_params)==error_endsit_params)
    ind_param_midsit_drop = mod(min_ind_midsit(1)-1,length(param_midsit_drop))+1;
    ind_param_midsit_window = mod(floor((min_ind_midsit(1)-1)/length(param_midsit_drop)),length(param_midsit_window))+1;
    ind_param_midsit_epsilon = mod(floor((min_ind_midsit(1)-1)/(length(param_midsit_drop)*length(param_midsit_window))),length(param_midsit_epsilon))+1;
    param_midsit_drop_best = param_midsit_drop(ind_param_midsit_drop)
    param_midsit_window_best = param_midsit_window(ind_param_midsit_window)
    param_midsit_epsilon_best = param_midsit_epsilon(ind_param_midsit_epsilon)

    min_ind_endsit = find(min(error_endsit_params)==error_endsit_params)
    ind_param_endsit_drop = mod(min_ind_endsit(1)-1,length(param_endsit_drop))+1;
    ind_param_endsit_window = mod(floor((min_ind_endsit(1)-1)/length(param_endsit_drop)),length(param_endsit_window))+1;
    ind_param_endsit_epsilon = mod(floor((min_ind_endsit(1)-1)/(length(param_endsit_drop)*length(param_endsit_window))),length(param_endsit_epsilon))+1;
    param_endsit_drop_best = param_endsit_drop(ind_param_endsit_drop)
    param_endsit_window_best = param_endsit_window(ind_param_endsit_window)
    param_endsit_epsilon_best = param_endsit_epsilon(ind_param_endsit_epsilon)


    temp_diff1 = []
    temp_diff2 = []
    temp_diff3 = []
    temp_diff4 = []
    temp_diff5 = []
    temp_diff6 = []
    temp_diff1_signed = []
    temp_diff2_signed = []
    temp_diff3_signed = []
    temp_diff4_signed = []
    temp_diff5_signed = []
    temp_diff6_signed = []

    
    
    temp_diff1k = []
    temp_diff2k = []
    temp_diff3k = []
    temp_diff4k = []
    temp_diff5k = []
    temp_diff6k = []
    for chosen_subject_trails_pairs = test_subjects
                        
        subject_id = floor(chosen_subject_trails_pairs/2);
        trial_number = mod(chosen_subject_trails_pairs,2)+type_trial; % + 4 for trials 4 and 5
           

            tot_drop = floor((max(yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(m:end))-(yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(1)))/param_stand_drop_best);
            stand_up_ind_rgb  = after_drop(-yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(m:end), tot_drop, param_stand_window_best,param_stand_epsilon_best) +m-1;

            flipped_zplotvals_pos_shoulders_smooth = flip(zplotvals_pos_shoulders_smooth{subject_id,trial_number-type_trial+1}(m:stand_up_ind_rgb));
            tot_drop = (max(flipped_zplotvals_pos_shoulders_smooth(m:stand_up_ind_rgb))-min(flipped_zplotvals_pos_shoulders_smooth(m:stand_up_ind_rgb)))/param_starttest_drop_best;        
            start_test  = before_drop(zplotvals_pos_shoulders_smooth{subject_id,trial_number-type_trial+1}(m:stand_up_ind_rgb), tot_drop, param_starttest_window_best,param_starttest_epsilon_best) +m-1;

            tot_drop = floor((max(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(stand_up_ind_rgb:end))-min(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(stand_up_ind_rgb:end)))/param_startturn_drop_best);
            reach_line_ind_rgb = before_drop(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(stand_up_ind_rgb:end),tot_drop,param_startturn_window_best,param_startturn_epsilon_best)+stand_up_ind_rgb-1;
            tot_drop = floor((max(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1})-min(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}))/param_midturn_drop_best);
            mid_turn = after_drop(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}, tot_drop, param_midturn_window_best,param_midturn_epsilon_best);

            tot_drop = floor((max(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(mid_turn:end))-min(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(mid_turn:end)))/param_endturn_drop_best);
            turn_complete_ind_rgb =  after_drop(-xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(mid_turn:end), tot_drop, param_endturn_window_best,param_endturn_epsilon_best)+mid_turn(1)-1;

            tot_drop = floor(((max(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(turn_complete_ind_rgb-1:end-50))-(min(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(turn_complete_ind_rgb-1:end-50))))/param_startsit_drop_best));

            rdy_tosit_rgb = before_drop(xplotvals_hip_absdiff_smooth{subject_id,trial_number-type_trial+1}(turn_complete_ind_rgb-1:end),tot_drop,param_startsit_window_best,param_startsit_epsilon_best)+turn_complete_ind_rgb-2;
            
            sit_pot_inds = floor(size(yplotvals_height_smooth{subject_id,trial_number-type_trial+1},2)/2);
            tot_drop = floor(((max(yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(sit_pot_inds:end))-(min(yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(sit_pot_inds:end))))/param_midsit_drop_best));
            bend_to_sit_rgb = after_drop(yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(sit_pot_inds:end),tot_drop,param_midsit_window_best,param_midsit_epsilon_best)+sit_pot_inds-1;
            tot_drop = floor(((max(yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(bend_to_sit_rgb:end))-(min(yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(bend_to_sit_rgb:end))))/param_endsit_drop_best));
            sit_rgb = after_drop(-yplotvals_height_smooth{subject_id,trial_number-type_trial+1}(bend_to_sit_rgb:end), tot_drop, param_endsit_window_best,param_endsit_epsilon_best) +bend_to_sit_rgb-1;

            disp('------Ground truth-----')
            disp(times_groundtruth{subject_id,trial_number-type_trial + 1})
            disp('------Results RGB-----')
            disp([start_test,stand_up_ind_rgb,reach_line_ind_rgb,turn_complete_ind_rgb,rdy_tosit_rgb,sit_rgb])
            disp('------Difference from ground truth (groundtruth-RGB) -----')
            diff_times = times_groundtruth{subject_id,trial_number-type_trial + 1}-[start_test,stand_up_ind_rgb,reach_line_ind_rgb,turn_complete_ind_rgb,rdy_tosit_rgb,sit_rgb];
            disp(diff_times)
            disp(['subject ',num2str(subject_id), ' trial ', num2str(trial_number)])

            temp_diff1(z) = abs(diff_times(1));
            temp_diff2(z)  = abs(diff_times(2));
            temp_diff3(z)  = abs(diff_times(3));
            temp_diff4(z)  = abs(diff_times(4));
            temp_diff5(z)  = abs(diff_times(5));
            temp_diff6(z)  = abs(diff_times(6));

            temp_diff1_signed(z) = (diff_times(1));
            temp_diff2_signed(z)  = (diff_times(2));
            temp_diff3_signed(z)  = (diff_times(3));
            temp_diff4_signed(z)  = (diff_times(4));
            temp_diff5_signed(z)  = (diff_times(5));
            temp_diff6_signed(z)  = (diff_times(6));

            rgb_vec = [start_test,stand_up_ind_rgb,reach_line_ind_rgb,turn_complete_ind_rgb,rdy_tosit_rgb,sit_rgb];
            z = z+1;

    end

	csvwrite(['subjects_1_5_cohort', num2str(cohort_ind),'.csv'],[training_subjects, test_subjects])
	csvwrite(['params_best_1_5_cohort',num2str(cohort_ind),'.csv'],[param_starttest_drop_best,param_starttest_window_best,param_starttest_epsilon_best,...
	param_stand_drop_best,param_stand_window_best,param_stand_epsilon_best,...
	param_startturn_drop_best,param_startturn_window_best,param_startturn_epsilon_best,...
	param_midturn_drop_best,param_midturn_window_best,param_midturn_epsilon_best,...
	param_endturn_drop_best,param_endturn_window_best,param_endturn_epsilon_best,...
	param_startsit_drop_best,param_startsit_window_best,param_startsit_epsilon_best,...
	param_midsit_drop_best,param_midsit_window_best,param_midsit_epsilon_best,...
	param_endsit_drop_best,param_endsit_window_best,param_endsit_epsilon_best])
	csvwrite(['errors_segments_signed_1_5_cohort', num2str(cohort_ind),'.csv'],[temp_diff1_signed;temp_diff2_signed;temp_diff3_signed;temp_diff4_signed;temp_diff5_signed;temp_diff6_signed])

	save(['params_search_space_1_5_cohort','.mat'],'param_starttest_drop','param_starttest_window',...
	'param_starttest_epsilon',...
	'param_stand_drop','param_stand_window','param_stand_epsilon',...
	'param_startturn_drop','param_startturn_window','param_startturn_epsilon',...
	'param_midturn_drop','param_midturn_window','param_midturn_epsilon',...
	'param_endturn_drop','param_endturn_window','param_endturn_epsilon',...
	'param_startsit_drop','param_startsit_window','param_startsit_epsilon',...
	'param_midsit_drop','param_midsit_window','param_midsit_epsilon',...
	'param_endsit_drop','param_endsit_window','param_endsit_epsilon')

	program_state = cohort_ind + 1;
end

timing_val = toc % Timing