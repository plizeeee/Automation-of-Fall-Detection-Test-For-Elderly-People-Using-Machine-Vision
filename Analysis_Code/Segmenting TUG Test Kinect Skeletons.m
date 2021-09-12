clear
clc

type_trial = 1; % 1 = trials 1,2  and 4 = trials 4,5

% After reducing the size of the script, avg time ~ 8 seconds per parameter

frame_rate = 28.3

% Search parameter space
% The space is explored seperately for each part of the TUG test, since they are independant of each other and performing the search independandly reduces the search time by several orders of magnitude

% Start test 3,3,8
param_starttest_drop = [1.05,1.09,1.12,1.17,1.22,1.3,1.4,1.55,1.7,2,2.5,3,3.5,4,5,6,7,8];
param_starttest_window = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
param_starttest_epsilon = [0,0.5,1,2,3,4,6,8,10,12,15,18,21,24,27,30,35,40,45,50];

% Stand 3,4,4
param_stand_drop = [1.05,1.09,1.12,1.17,1.22,1.3,1.4,1.55,1.7,2,2.5,3,3.5,4,5,6,7,8];
param_stand_window = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
param_stand_epsilon = [0,0.5,1,2,3,4,6,8,10,12,15,18,21,24,27,30,35,40,45,50];

% Start Turn 1.7,5,3
param_startturn_drop = [1.05,1.09,1.12,1.17,1.22,1.3,1.4,1.55,1.7,2,2.5,3,3.5,4,5,6,7,8];
param_startturn_window = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
param_startturn_epsilon = [0,0.5,1,2,3,4,6,8,10,12,15,18,21,24,27,30,35,40,45,50];

% End Turn  mid =() end = (1.8,3,4)
param_midturn_drop = [1.05,1.09,1.12,1.17,1.22,1.3,1.4,1.55,1.7,2,2.5,3,3.5,4,5,6,7,8];
param_midturn_window = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
param_midturn_epsilon = [0,0.5,1,2,3,4,6,8,10,12,15,18,21,24,27,30,35,40,45,50];
param_endturn_drop = [1.05,1.09,1.12,1.17,1.22,1.3,1.4,1.55,1.7,2,2.5,3,3.5,4,5,6,7,8];
param_endturn_window = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
param_endturn_epsilon = [0,0.5,1,2,3,4,6,8,10,12,15,18,21,24,27,30,35,40,45,50];

% Start Sit: 1.5,3,4
param_startsit_drop = [1.05,1.09,1.12,1.17,1.22,1.3,1.4,1.55,1.7,2,2.5,3,3.5,4,5,6,7,8];
param_startsit_window = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
param_startsit_epsilon = [0,0.5,1,2,3,4,6,8,10,12,15,18,21,24,27,30,35,40,45,50];

% End Sit: mid = (1.4,5,0) end = (1.4,7,4)
param_midsit_drop = [1.05,1.09,1.12,1.17,1.22,1.3,1.4,1.55,1.7,2,2.5,3,3.5,4,5,6,7,8];
param_midsit_window = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
param_midsit_epsilon = [0,0.5,1,2,3,4,6,8,10,12,15,18,21,24,27,30,35,40,45,50];
param_endsit_drop = [1.05,1.09,1.12,1.17,1.22,1.3,1.4,1.55,1.7,2,2.5,3,3.5,4,5,6,7,8];
param_endsit_window = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
param_endsit_epsilon = [0,0.5,1,2,3,4,6,8,10,12,15,18,21,24,27,30,35,40,45,50];

tic
program_state = 1
      % If program crashes, load most recent state
% load('cohort_most_recent.mat')

%%%%%%%%%%%%%% Pre loading values for faster computation


yplotvals_heightk_smooth = cell(30,2);
zplotvals_pelvis_global_smoothk = cell(30,2);
yplotvals_pelvis_smoothk = cell(30,2);
xplotvals_hip_absdiff_smoothk = cell(30,2);

zplotvals_pelvis_global_smoothk_full = cell(30,2);
xplotvals_hip_absdiff_smoothk_full = cell(30,2);
zplotvals_shoulder_globalk_smooth = cell(30,2);
times_groundtruth = cell(30,2)



fail_ind = 1 % Number of failed Kinect skeletons (Since the Kinect Skeletons did not work 100 percent of the time)
subj_trial = 10
zk = 1;
for subject_id = 1:30
    for trial_number = type_trial:type_trial+1
		keypts_1Darr_kinect_full = csvread(['output/subjects/subject', num2str(subject_id), '/trial', num2str(trial_number),'/kinect_skeleton.csv']);
		cut_times = csvread(['output/subjects/subject', num2str(subject_id), '/trial', num2str(trial_number),'/cut_times.csv']);
		times_groundtruth{subject_id,trial_number-type_trial + 1} = csvread(['output/subjects/subject',num2str(subject_id), '/trial', num2str(trial_number),'/segmented_times_mean.csv']);
		ind_start = find(keypts_1Darr_kinect_full(:,47) == cut_times(1));
		ind_end = find(keypts_1Darr_kinect_full(:,47) == cut_times(2));
		
		% These are trials where Kinect Failed, so the segmentation algorithm will not work.
		if (isempty(ind_start)||isempty(ind_end)) || (subject_id == 14) || (subject_id == 26 && trial_number ==2) || (trial_number==4) ||(trial_number==5)|| (subject_id == 9 &&trial_number==2)
			  id_trial{fail_ind} = [subject_id,trial_number];
			fail_ind = fail_ind + 1;
		else
		keypts_1Darr_kinect = keypts_1Darr_kinect_full(ind_start-1:ind_end-1,:);

		nk = size(keypts_1Darr_kinect_full,1);

		smoothed_rfoot = smoothdata(keypts_1Darr_kinect(:,12),'movmedian',5);
		smoothed_lfoot = smoothdata(keypts_1Darr_kinect(:,21),'movmedian',5);

		zplotvals_pelvis_globalk_full = zeros(1,nk);
		xplotvals_hip_absdiffk_full = zeros(1,nk);



		for i = 1:nk
			zplotvals_pelvis_globalk_full(i) = (keypts_1Darr_kinect_full(i,6)+keypts_1Darr_kinect_full(i,15))/2;
			xplotvals_hip_absdiffk_full(i) = abs(keypts_1Darr_kinect_full(i,40)-keypts_1Darr_kinect_full(i,31));
		end


		nk = size(keypts_1Darr_kinect,1);
		yplotvals_heightk = zeros(1,nk);
		xplotvals_leftsk = zeros(1,nk);
		xplotvals_rightsk = zeros(1,nk);
		yplotvals_pelvisk = zeros(1,nk);
		yplotvals_hipsk = zeros(1,nk);
		yplotvals_hip_knee_absdiffk = zeros(1,nk);
		zplotvals_pelvis_globalk = zeros(1,nk);
		xplotvals_hip_absdiffk = zeros(1,nk); 
		zplotvals_shoulder_globalk = zeros(1,nk);
		for i = 1:nk
			% Relative positions
			yplotvals_heightk(i) = keypts_1Darr_kinect(i,26);
			xplotvals_leftsk(i) = keypts_1Darr_kinect(i,28);
			xplotvals_rightsk(i) = keypts_1Darr_kinect(i,37);
			yplotvals_pelvisk(i) = (smoothed_rfoot(i)-smoothed_lfoot(i));
			yplotvals_hipsk(i) = (keypts_1Darr_kinect(i,5)+keypts_1Darr_kinect(i,14))/2;   
			yplotvals_hip_knee_absdiffk(i) = abs(keypts_1Darr_kinect(i,5)-keypts_1Darr_kinect(i,8))+abs(keypts_1Darr_kinect(i,14)-keypts_1Darr_kinect(i,17));
			% The z value is actually the depth of the image, it's just represented differently for the pinhole camera model 
			% Hip gloval y pos over time
			zplotvals_pelvis_globalk(i) = (keypts_1Darr_kinect(i,6)+keypts_1Darr_kinect(i,15))/2;
			zplotvals_shoulder_globalk(i) = (keypts_1Darr_kinect(i,30) + keypts_1Darr_kinect(i,39))/2;
			xplotvals_hip_absdiffk(i) = abs(keypts_1Darr_kinect(i,13)-keypts_1Darr_kinect(i,4));                      
		end

		okk = 1:size(keypts_1Darr_kinect,1);
		yplotvals_heightk_smooth{subject_id,trial_number} = smoothdata(yplotvals_heightk,'movmedian',8);
		zplotvals_pelvis_global_smoothk{subject_id,trial_number} = smoothdata(zplotvals_pelvis_globalk,'movmedian',8);
		yplotvals_pelvis_smoothk{subject_id,trial_number} = smoothdata(yplotvals_pelvisk,'movmedian',8);
		xplotvals_hip_absdiff_smoothk{subject_id,trial_number} =  smoothdata(xplotvals_hip_absdiffk,'movmedian',8);

		zplotvals_pelvis_global_smoothk_full{subject_id,trial_number} = smoothdata(zplotvals_pelvis_globalk_full,'movmedian',8);
		xplotvals_hip_absdiff_smoothk_full{subject_id,trial_number} =  smoothdata(xplotvals_hip_absdiffk_full,'movmedian',8);
		zplotvals_shoulder_globalk_smooth{subject_id,trial_number} =  smoothdata(zplotvals_shoulder_globalk,'movmedian',8);
		subj_trial_no_error(zk) = subj_trial;
		zk = zk + 1;
		end
		subj_trial = subj_trial + 1;

    end
end
rand_order = randperm(50);
cohort = cell(5,1)
cohort{1} = subj_trial_no_error(rand_order(1:10))
cohort{2} = subj_trial_no_error(rand_order(11:20))
cohort{3} = subj_trial_no_error(rand_order(21:30))
cohort{4} = subj_trial_no_error(rand_order(31:40))
cohort{5} = subj_trial_no_error(rand_order(41:50))

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
                fail_ind = 1;
                for chosen_subject_trails_pairs = training_subjects
                        
                    subject_id = floor(chosen_subject_trails_pairs/2);
                    trial_number = mod(chosen_subject_trails_pairs,2)+type_trial; % + 1 for trials 1 and 2            
                        m = 1; % Start frame (start will normally just be the first frame of the video, since user will be seated

                        tot_drop = floor((max(yplotvals_heightk_smooth{subject_id,trial_number}(m:end))-(yplotvals_heightk_smooth{subject_id,trial_number}(1)))/param_stand_drop(drop_ind));
                        stand_up_ind_k  = before_drop(-yplotvals_heightk_smooth{subject_id,trial_number}(m:end), tot_drop, param_stand_window(window_ind),param_stand_epsilon(epsilon_ind)) +m-1;
                        
						flipped_zplotvals_pos_shoulders_smooth_k = flip(zplotvals_shoulder_globalk_smooth{subject_id,trial_number}(m:times_groundtruth{subject_id,trial_number-type_trial+1}(2)));
                        tot_drop = (max(zplotvals_shoulder_globalk_smooth{subject_id,trial_number}(m:times_groundtruth{subject_id,trial_number-type_trial+1}(2)))-min(zplotvals_shoulder_globalk_smooth{subject_id,trial_number}(m:times_groundtruth{subject_id,trial_number-type_trial+1}(2))))/param_starttest_drop(drop_ind);        
                        start_test_k  = before_drop(zplotvals_shoulder_globalk_smooth{subject_id,trial_number}(m:times_groundtruth{subject_id,trial_number-type_trial+1}(2)), tot_drop, param_starttest_window(window_ind),param_starttest_epsilon(epsilon_ind)) +m-1;

                        ind_endrl = floor(0.6*size(xplotvals_hip_absdiff_smoothk{subject_id,trial_number},2));
                        tot_drop = floor((max(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(times_groundtruth{subject_id,trial_number-type_trial+1}(2):ind_endrl))-min(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(times_groundtruth{subject_id,trial_number-type_trial+1}(2):ind_endrl)))/param_startturn_drop(drop_ind));
                        reach_line_ind_k = before_drop(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(times_groundtruth{subject_id,trial_number-type_trial+1}(2):end),tot_drop,param_startturn_window(window_ind),param_startturn_epsilon(epsilon_ind))+times_groundtruth{subject_id,trial_number-type_trial+1}(2)-1;

                        ind_endtn = floor(0.7*size(xplotvals_hip_absdiff_smoothk{subject_id,trial_number},2));
                        ind_starttn = floor(0.32*size(xplotvals_hip_absdiff_smoothk{subject_id,trial_number},2)); % (1.8,4,2.5),(1.8,3,3)
                        tot_drop = floor((max(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(ind_starttn:ind_endtn))-min(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(ind_starttn:ind_endtn)))/param_midturn_drop(drop_ind));
                        mid_turn = after_drop(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(ind_starttn:ind_endtn), tot_drop, param_midturn_window(window_ind),param_midturn_epsilon(epsilon_ind))+ind_starttn(1)-1;

                        tot_drop = floor((max(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(mid_turn:ind_endtn))-min(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(mid_turn:ind_endtn)))/param_endturn_drop(drop_ind));
                        turn_complete_ind_k =  after_drop(-xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(mid_turn:ind_endtn), tot_drop, param_endturn_window(window_ind),param_endturn_epsilon(epsilon_ind))+mid_turn(1)-1;

                        tot_drop = floor(((max(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(times_groundtruth{subject_id,trial_number-type_trial+1}(4)-1:end-50))-(min(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(times_groundtruth{subject_id,trial_number-type_trial+1}(4)-1:end-50))))/param_startsit_drop(drop_ind)));
            

                        rdy_tosit_k = before_drop(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(times_groundtruth{subject_id,trial_number-type_trial+1}(4)-1:end),tot_drop,param_startsit_window(window_ind),param_startsit_epsilon(epsilon_ind))+times_groundtruth{subject_id,trial_number-type_trial+1}(4)-2;

                        sit_pot_inds = floor(size(yplotvals_heightk_smooth{subject_id,trial_number},2)/2); %(1.4,5,0),(1.4,7,4)
                        tot_drop = floor(((max(yplotvals_heightk_smooth{subject_id,trial_number}(sit_pot_inds:end))-(min(yplotvals_heightk_smooth{subject_id,trial_number}(sit_pot_inds:end))))/param_midsit_drop(drop_ind)));
                        bend_to_sit_k = after_drop(yplotvals_heightk_smooth{subject_id,trial_number}(sit_pot_inds:end),tot_drop,param_midsit_window(window_ind),param_midsit_epsilon(epsilon_ind))+sit_pot_inds-1;
                        tot_drop = floor(((max(yplotvals_heightk_smooth{subject_id,trial_number}(bend_to_sit_k:end))-(min(yplotvals_heightk_smooth{subject_id,trial_number}(bend_to_sit_k:end))))/param_endsit_drop(drop_ind)));
                        sit_k = after_drop(-yplotvals_heightk_smooth{subject_id,trial_number}(bend_to_sit_k:end), tot_drop, param_endsit_window(window_ind),param_endsit_epsilon(epsilon_ind)) +bend_to_sit_k-1;
       
                        diff_times = times_groundtruth{subject_id,trial_number-type_trial+1}-[start_test_k,stand_up_ind_k,reach_line_ind_k,turn_complete_ind_k,rdy_tosit_k,sit_k];
                        temp_diff1k(zk) = abs(diff_times(1));
                        temp_diff2k(zk)  = abs(diff_times(2));
                        temp_diff3k(zk)  = abs(diff_times(3));
                        temp_diff4k(zk)  = abs(diff_times(4));
                        temp_diff5k(zk)  = abs(diff_times(5));
                        temp_diff6k(zk)  = abs(diff_times(6));

                        temp_diff1k_signed(zk) = (diff_times(1));
                        temp_diff2k_signed(zk)  = (diff_times(2));
                        temp_diff3k_signed(zk)  = (diff_times(3));
                        temp_diff4k_signed(zk)  = (diff_times(4));
                        temp_diff5k_signed(zk)  = (diff_times(5));
                        temp_diff6k_signed(zk)  = (diff_times(6));
                        zk = zk+1;
                      
            end
                disp(['param index = ' , num2str(param_ind)])
                error_starttest_params(param_ind) = mean(temp_diff1k)/frame_rate;
                error_stand_params(param_ind) = mean(temp_diff2k)/frame_rate;
                error_startturn_params(param_ind) = mean(temp_diff3k)/frame_rate;
                error_endturn_params(param_ind) = mean(temp_diff4k)/frame_rate;
                error_startsit_params(param_ind) = mean(temp_diff5k)/frame_rate;
                error_endsit_params(param_ind) = mean(temp_diff6k)/frame_rate;
        
            end 
        end
    end
	
    z = 1;
    zk = 1;
    fail_ind = 1;
	
    err = min(error_starttest_params)+ min(error_stand_params) + min(error_startturn_params) +...
        min(error_endturn_params) + min(error_startsit_params) + min(error_endsit_params)

    min_ind_starttest = find(min(error_starttest_params)==error_starttest_params)
    ind_param_starttest_drop = mod(min_ind_starttest(1)-1,length(param_starttest_drop))+1;
    ind_param_starttest_window = mod(floor((min_ind_starttest(1)-1)/length(param_starttest_drop)),length(param_starttest_window))+1;
    ind_param_starttest_epsilon = mod(floor((min_ind_starttest(1)-1)/(length(param_starttest_drop)*length(param_starttest_window))),length(param_starttest_epsilon))+1;
    param_starttest_drop_best = param_starttest_drop(ind_param_starttest_drop)
    param_starttest_window_best = param_starttest_window(ind_param_starttest_window)
    param_starttest_epsilon_best = param_starttest_epsilon(ind_param_starttest_epsilon)

    min_ind_stand = find(min(error_stand_params)==error_stand_params)
    ind_param_stand_drop = mod(min_ind_stand(1)-1,length(param_stand_drop))+1;
    ind_param_stand_window = mod(floor((min_ind_stand(1)-1)/length(param_stand_drop)),length(param_stand_window))+1;
    ind_param_stand_epsilon = mod(floor((min_ind_stand(1)-1)/(length(param_stand_drop)*length(param_stand_window))),length(param_stand_epsilon))+1;
    param_stand_drop_best = param_stand_drop(ind_param_stand_drop)
    param_stand_window_best = param_stand_window(ind_param_stand_window)
    param_stand_epsilon_best = param_stand_epsilon(ind_param_stand_epsilon)

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
    temp_diff1k_signed = []
    temp_diff2k_signed = []
    temp_diff3k_signed = []
    temp_diff4k_signed = []
    temp_diff5k_signed = []
    temp_diff6k_signed = []
	
    for chosen_subject_trails_pairs = test_subjects
                        
        subject_id = floor(chosen_subject_trails_pairs/2);
        trial_number = mod(chosen_subject_trails_pairs,2)+type_trial; % + 4 for trials 4 and 5
                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		m = 1; % Start frame (start will normally just be the first frame of the video, since user will be seated

		tot_drop = floor((max(yplotvals_heightk_smooth{subject_id,trial_number}(m:end))-(yplotvals_heightk_smooth{subject_id,trial_number}(1)))/param_stand_drop_best);
		stand_up_ind_k  = before_drop(-yplotvals_heightk_smooth{subject_id,trial_number}(m:end), tot_drop, param_stand_window_best,param_stand_epsilon_best) +m-1;

		flipped_zplotvals_pos_shoulders_smooth_k = flip(zplotvals_shoulder_globalk_smooth{subject_id,trial_number}(m:stand_up_ind_k));
		tot_drop = (max(zplotvals_shoulder_globalk_smooth{subject_id,trial_number}(m:stand_up_ind_k))-min(zplotvals_shoulder_globalk_smooth{subject_id,trial_number}(m:stand_up_ind_k)))/param_starttest_drop_best;        
		start_test_k  = before_drop(zplotvals_shoulder_globalk_smooth{subject_id,trial_number}(m:stand_up_ind_k), tot_drop, param_starttest_window_best,param_starttest_epsilon_best) +m-1;

		ind_endrl = floor(0.6*size(xplotvals_hip_absdiff_smoothk{subject_id,trial_number},2));
		tot_drop = floor((max(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(min([stand_up_ind_k,ind_endrl-1]):ind_endrl))-min(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(min([stand_up_ind_k,ind_endrl-1]):ind_endrl)))/param_startturn_drop_best);
		reach_line_ind_k = before_drop(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(stand_up_ind_k:end),tot_drop,param_startturn_window_best,param_startturn_epsilon_best)+stand_up_ind_k-1;

		ind_endtn = floor(0.7*size(xplotvals_hip_absdiff_smoothk{subject_id,trial_number},2));
		ind_starttn = floor(0.32*size(xplotvals_hip_absdiff_smoothk{subject_id,trial_number},2)); % (1.8,4,2.5),(1.8,3,3)
		tot_drop = floor((max(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(ind_starttn:ind_endtn))-min(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(ind_starttn:ind_endtn)))/param_midturn_drop_best);
		mid_turn = after_drop(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(ind_starttn:ind_endtn), tot_drop, param_midturn_window_best,param_midturn_epsilon_best)+ind_starttn(1)-1;

		tot_drop = floor((max(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(mid_turn:ind_endtn))-min(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(mid_turn:ind_endtn)))/param_endturn_drop_best);
		turn_complete_ind_k =  after_drop(-xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(mid_turn:ind_endtn), tot_drop, param_endturn_window_best,param_endturn_epsilon_best)+mid_turn(1)-1;
		tot_drop = floor(((max(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(turn_complete_ind_k-2:end-50))-(min(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(turn_complete_ind_k-2:end-50))))/param_startsit_drop_best)); 

		rdy_tosit_k = before_drop(xplotvals_hip_absdiff_smoothk{subject_id,trial_number}(turn_complete_ind_k-2:end),tot_drop,param_startsit_window_best,param_startsit_epsilon_best)+turn_complete_ind_k-3;

		sit_pot_inds = floor(size(yplotvals_heightk_smooth{subject_id,trial_number},2)/2); %(1.4,5,0),(1.4,7,4)
		tot_drop = floor(((max(yplotvals_heightk_smooth{subject_id,trial_number}(sit_pot_inds:end))-(min(yplotvals_heightk_smooth{subject_id,trial_number}(sit_pot_inds:end))))/param_midsit_drop_best));
		bend_to_sit_k = after_drop(yplotvals_heightk_smooth{subject_id,trial_number}(sit_pot_inds:end),tot_drop,param_midsit_window_best,param_midsit_epsilon_best)+sit_pot_inds-1;
		tot_drop = floor(((max(yplotvals_heightk_smooth{subject_id,trial_number}(bend_to_sit_k:end))-(min(yplotvals_heightk_smooth{subject_id,trial_number}(bend_to_sit_k:end))))/param_endsit_drop_best));
		sit_k = after_drop(-yplotvals_heightk_smooth{subject_id,trial_number}(bend_to_sit_k:end), tot_drop, param_endsit_window_best,param_endsit_epsilon_best) +bend_to_sit_k-1;

		diff_times = times_groundtruth{subject_id,trial_number-type_trial+1}-[start_test_k,stand_up_ind_k,reach_line_ind_k,turn_complete_ind_k,rdy_tosit_k,sit_k];
		temp_diff1k(zk) = abs(diff_times(1));
		temp_diff2k(zk)  = abs(diff_times(2));
		temp_diff3k(zk)  = abs(diff_times(3));
		temp_diff4k(zk)  = abs(diff_times(4));
		temp_diff5k(zk)  = abs(diff_times(5));
		temp_diff6k(zk)  = abs(diff_times(6));

		temp_diff1k_signed(zk) = (diff_times(1));
		temp_diff2k_signed(zk)  = (diff_times(2));
		temp_diff3k_signed(zk)  = (diff_times(3));
		temp_diff4k_signed(zk)  = (diff_times(4));
		temp_diff5k_signed(zk)  = (diff_times(5));
		temp_diff6k_signed(zk)  = (diff_times(6));
		zk = zk+1;

    end
  
	csvwrite(['subjects_1_5_kinect_cohortB', num2str(cohort_ind),'.csv'],[training_subjects, test_subjects])
	csvwrite(['params_best_1_5_kinect_cohortB',num2str(cohort_ind),'.csv'],[param_starttest_drop_best,param_starttest_window_best,param_starttest_epsilon_best,...
	param_stand_drop_best,param_stand_window_best,param_stand_epsilon_best,...
	param_startturn_drop_best,param_startturn_window_best,param_startturn_epsilon_best,...
	param_midturn_drop_best,param_midturn_window_best,param_midturn_epsilon_best,...
	param_endturn_drop_best,param_endturn_window_best,param_endturn_epsilon_best,...
	param_startsit_drop_best,param_startsit_window_best,param_startsit_epsilon_best,...
	param_midsit_drop_best,param_midsit_window_best,param_midsit_epsilon_best,...
	param_endsit_drop_best,param_endsit_window_best,param_endsit_epsilon_best])
	csvwrite(['errors_segments_signed_1_5_kinect_cohortB', num2str(cohort_ind),'.csv'],[temp_diff1k_signed;temp_diff2k_signed;temp_diff3k_signed;temp_diff4k_signed;temp_diff5k_signed;temp_diff6k_signed])

	save(['params_search_space_1_5_kinect_cohortB','.mat'],'param_starttest_drop','param_starttest_window',...
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

time_val = toc


