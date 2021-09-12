clear
clc
close all
frame_rate = 28.3;
num_labelers = 5

for labeler_num = 1:num_labelers
    z = 1;
    for subject_id = 1:30
        for trial_number = [1,2,4,5]
            times = csvread(['output/subjects/subject',num2str(subject_id), '/trial', num2str(trial_number),'/segmented_times', num2str(labeler_num), '.csv']);
            start_stand(z) = times(1)
            finish_stand(z) = times(2)
            reach_line(z) = times(3)
            finish_turn(z) = times(4)
            reach_chair(z) = times(5)
            sit(z) = times(6)
            z = z+1;
        end
    end
    start_stand_all_labelers{labeler_num} = start_stand
    finish_stand_all_labelers{labeler_num} = finish_stand
    reach_line_all_labelers{labeler_num} = reach_line
    finish_turn_all_labelers{labeler_num} = finish_turn
    reach_chair_all_labelers{labeler_num} = reach_chair
    sit_all_labelers{labeler_num} = sit
end

figure(1)
hold on
labeler_id = cell(1,labeler_num);
for labeler_num = 1:num_labelers
   plot(1:z-1,start_stand_all_labelers{labeler_num}, 'o') 
      labeler_id{labeler_num} = ['labeler',num2str(labeler_num)]; 
   title('start stand all labelers')
end
   legend(labeler_id)
   
for ind = 1:z-1
    for labeler_num = 1:num_labelers
        temp(labeler_num) = start_stand_all_labelers{labeler_num}(ind);
    end
%     meanval(z) = mean(temp);
    maxval(ind) = max(temp);
    minval(ind) = min(temp);
    tot_dev(ind) = maxval(ind)-minval(ind);
    mean_label_trial1 = mean(temp);
    diff_mean = temp-mean_label_trial1;
    mean_label_trial(ind) = mean(abs(diff_mean));
end
disp('----------start stand all labelers--------------')
mean_dev = mean(tot_dev)/frame_rate
std_dev = std(tot_dev)/frame_rate
max_dev = max(tot_dev)/frame_rate
ind  = find(round(max_dev*frame_rate)==tot_dev)
mean_mean_diff_gt = mean(mean_label_trial)/frame_rate
std_mean_diff_gt = std(mean_label_trial)/frame_rate


figure(2)
hold on
for labeler_num = 1:num_labelers
   plot(1:z-1,finish_stand_all_labelers{labeler_num}, 'o') 
   title('finish stand all labelers')
end
   legend(labeler_id)


for ind = 1:z-1
    for labeler_num = 1:num_labelers
        temp(labeler_num) = finish_stand_all_labelers{labeler_num}(ind);
    end
%     meanval(z) = mean(temp);
    maxval(ind) = max(temp);
    minval(ind) = min(temp);
    tot_dev(ind) = maxval(ind)-minval(ind);
    mean_label_trial1 = mean(temp);
    diff_mean = temp-mean_label_trial1;
    mean_label_trial(ind) = mean(abs(diff_mean));
end
disp('----------finish stand all labelers--------------')
mean_dev = mean(tot_dev)/frame_rate
std_dev = std(tot_dev)/frame_rate
max_dev = max(tot_dev)/frame_rate
ind  = find(round(max_dev*frame_rate)==tot_dev)
mean_mean_diff_gt = mean(mean_label_trial)/frame_rate
std_mean_diff_gt = std(mean_label_trial)/frame_rate


figure(3)
hold on 

for labeler_num = 1:num_labelers
   plot(1:z-1,reach_line_all_labelers{labeler_num}, 'o') 
   title('reach line all labelers')
end
   legend(labeler_id)
for ind = 1:z-1
    for labeler_num = 1:num_labelers
        temp(labeler_num) = reach_line_all_labelers{labeler_num}(ind);
    end
%     meanval(z) = mean(temp);
    maxval(ind) = max(temp);
    minval(ind) = min(temp);
    mean_label_trial1 = mean(temp);
    tot_dev(ind) = maxval(ind)-minval(ind);
    diff_mean = temp-mean_label_trial1;
    mean_label_trial(ind) = mean(abs(diff_mean));
end
disp('----------reach line all labelers--------------')
mean_dev = mean(tot_dev)/frame_rate
std_dev = std(tot_dev)/frame_rate
max_dev = max(tot_dev)/frame_rate
ind  = find(round(max_dev*frame_rate)==tot_dev)
mean_mean_diff_gt = mean(mean_label_trial)/frame_rate
std_mean_diff_gt = std(mean_label_trial)/frame_rate


figure(4)
hold on 
for labeler_num = 1:num_labelers
   plot(1:z-1,finish_turn_all_labelers{labeler_num}, 'o') 
   title('finish turn all labelers')
end
   legend(labeler_id)
for ind = 1:z-1
    for labeler_num = 1:num_labelers
        temp(labeler_num) = finish_turn_all_labelers{labeler_num}(ind);
    end
%     meanval(z) = mean(temp);
    maxval(ind) = max(temp);
    minval(ind) = min(temp);
    tot_dev(ind) = maxval(ind)-minval(ind);
    mean_label_trial1 = mean(temp);
    diff_mean = temp-mean_label_trial1;
    mean_label_trial(ind) = mean(abs(diff_mean));
end
disp('----------finish turn all labelers--------------')
mean_dev = mean(tot_dev)/frame_rate
std_dev = std(tot_dev)/frame_rate
max_dev = max(tot_dev)/frame_rate
ind  = find(round(max_dev*frame_rate)==tot_dev)
mean_mean_diff_gt = mean(mean_label_trial)/frame_rate
std_mean_diff_gt = std(mean_label_trial)/frame_rate




figure(5)
hold on 
for labeler_num = 1:num_labelers
   plot(1:z-1,reach_chair_all_labelers{labeler_num}, 'o') 
   title('reach chair all labelers')
end
   legend(labeler_id)
for ind = 1:z-1
    for labeler_num = 1:num_labelers
        temp(labeler_num) = reach_chair_all_labelers{labeler_num}(ind);
    end
%     meanval(z) = mean(temp);
    maxval(ind) = max(temp);
    minval(ind) = min(temp);
    tot_dev(ind) = maxval(ind)-minval(ind);
    mean_label_trial1 = mean(temp);
    diff_mean = temp-mean_label_trial1;
    mean_label_trial(ind) = mean(abs(diff_mean));
end
disp('----------reach chair all labelers--------------')
mean_dev = mean(tot_dev)/frame_rate/frame_rate
std_dev = std(tot_dev)/frame_rate/frame_rate
max_dev = max(tot_dev)/frame_rate/frame_rate
ind  = find(round(max_dev*frame_rate)==tot_dev)
mean_mean_diff_gt = mean(mean_label_trial)/frame_rate
std_mean_diff_gt = std(mean_label_trial)/frame_rate
figure(6)
hold on 
for labeler_num = 1:num_labelers
   plot(1:z-1,sit_all_labelers{labeler_num}, 'o')
   title('sit all labelers')
end
   legend(labeler_id)
for ind = 1:z-1
    for labeler_num = 1:num_labelers
        temp(labeler_num) = sit_all_labelers{labeler_num}(ind);
    end
%     meanval(z) = mean(temp);
    maxval(ind) = max(temp);
    minval(ind) = min(temp);
    tot_dev(ind) = maxval(ind)-minval(ind);
    mean_label_trial1 = mean(temp);
    diff_mean = temp-mean_label_trial1;
    mean_label_trial(ind) = mean(abs(diff_mean));
end
disp('----------sit all labelers--------------')
mean_dev = mean(tot_dev)/frame_rate
std_dev = std(tot_dev)/frame_rate
max_dev = max(tot_dev)/frame_rate
ind  = find(round(max_dev*frame_rate)==tot_dev)
mean_mean_diff_gt = mean(mean_label_trial)/frame_rate
std_mean_diff_gt = std(mean_label_trial)/frame_rate



%%
for subject_id = 1:30
   for trial_number = [1,2,4,5]
       for labeler_num = 1:num_labelers
           times(labeler_num,:) = csvread(['output/subjects/subject',num2str(subject_id), '/trial', num2str(trial_number),'/segmented_times', num2str(labeler_num), '.csv'])
       end
       segmented_times_mean = round(mean(times))
       csvwrite(['output/subjects/subject',num2str(subject_id), '/trial', num2str(trial_number),'/segmented_times_mean.csv'],segmented_times_mean)
   end
end