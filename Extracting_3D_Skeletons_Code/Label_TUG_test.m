person_number = 1
trial_start = 4


labeler_num = 5

 
ptime = 0.001
if trial_start == 1
    vec_trials = [1,2,4,5]
elseif trial_start == 2
    vec_trials = [2,4,5]
elseif trial_start == 4
    vec_trials = [4,5]
elseif trial_start == 5
    vec_trials = 5
end
    
subject_id = person_number
for trial_number = vec_trials
    obj = VideoReader(['output/subjects/subject',num2str(subject_id), '/trial', num2str(trial_number),'/color_output_kinect.avi']);
    FS = stoploop({'(1)' 'Wait till click for', 'Standing Phase'}) ;
    k = 1;
    f = figure(2);
    this_frame = readFrame(obj);
    imh = imshow(this_frame);
    x0=50;
    y0=10;
    width=800;
    height=600;
    disp(['Subject number is: ', num2str(subject_id) ' Trial number is: ', num2str(trial_number)])
    set(gcf,'units','points','position',[x0,y0,width,height])
%     tic
        while ~FS.Stop() && hasFrame(obj)
            this_frame = readFrame(obj);
            pause(ptime)
            set(imh, 'CData', this_frame);
            drawnow();
            title(['frame ',num2str(k)])
            k = k+1;
        end
        frame_rdy_stand = k

        FS = stoploop({'(2)' 'Wait till click for', 'Standing'}) ;
        while ~FS.Stop() && hasFrame(obj)
            this_frame = readFrame(obj);
            pause(ptime)
            set(imh, 'CData', this_frame);
            drawnow();
            title(['frame ',num2str(k)])
            k = k+1;
        end
        frame_stand = k
        
        FS = stoploop({'(3)' 'Wait till click for', 'Reaches Line'}) ;
        while ~FS.Stop() && hasFrame(obj)
            this_frame = readFrame(obj);
            pause(ptime)
            set(imh, 'CData', this_frame);
            drawnow();
            title(['frame ',num2str(k)])
            k = k+1;
        end
        frame_line = k

        FS = stoploop({'(4)' 'Wait till click for', 'Turn Complete'}) ;
        while ~FS.Stop() && hasFrame(obj)
            this_frame = readFrame(obj);
            pause(ptime)
            set(imh, 'CData', this_frame);
            drawnow();
            title(['frame ',num2str(k)])
            k = k+1;
        end
        frame_finish_turn = k

        FS = stoploop({'(5)' 'Wait till click for', 'Reach chair'}) ;
        while ~FS.Stop() && hasFrame(obj)
            this_frame = readFrame(obj);
            pause(ptime)
            set(imh, 'CData', this_frame);
            drawnow();
            title(['frame ',num2str(k)])
            k = k+1;
        end
        frame_reach_chair = k
                
        FS = stoploop({'(6)' 'Wait till click for', 'Sit'}) ;
        while ~FS.Stop() && hasFrame(obj)
            this_frame = readFrame(obj);
            pause(ptime)
            set(imh, 'CData', this_frame);
            drawnow();
            title(['frame ',num2str(k)])
            k = k+1;
        end
        frame_sit = k

        time_vec = [frame_rdy_stand,frame_stand,frame_line,frame_finish_turn,frame_reach_chair,frame_sit]
%         toc
    %%% here
    csvwrite(['output/subjects/subject',num2str(subject_id), '/trial', num2str(trial_number),'/segmented_times', num2str(labeler_num), '.csv'],time_vec)
    pause
end

for subject_id = person_number+1:30
    for trial_number = [1,2,4,5]
        obj = VideoReader(['output/subjects/subject',num2str(subject_id), '/trial', num2str(trial_number),'/color_output_kinect.avi']);
        FS = stoploop({'(1)' 'Wait till click for', 'Starting to stand'}) ;
        k = 1;
        f = figure(2);
        this_frame = readFrame(obj);
        imh = imshow(this_frame);
        x0=50;
        y0=10;
        width=800;
        height=600;
        disp(['Subject number is: ', num2str(subject_id) ' Trial number is: ', num2str(trial_number)])
        set(gcf,'units','points','position',[x0,y0,width,height]);
%         tic
        while ~FS.Stop() && hasFrame(obj)
            this_frame = readFrame(obj);
            pause(ptime)
            set(imh, 'CData', this_frame);
            drawnow();
            title(['frame ',num2str(k)])
            k = k+1;
        end
        frame_rdy_stand = k

        FS = stoploop({'(2)' 'Wait till click for', 'Standing'}) ;
        while ~FS.Stop() && hasFrame(obj)
            this_frame = readFrame(obj);
            pause(ptime)
            set(imh, 'CData', this_frame);
            drawnow();
            title(['frame ',num2str(k)])
            k = k+1;
        end
        frame_stand = k
        
        FS = stoploop({'(3)' 'Wait till click for', 'Reaches Line'}) ;
        while ~FS.Stop() && hasFrame(obj)
            this_frame = readFrame(obj);
            pause(ptime)
            set(imh, 'CData', this_frame);
            drawnow();
            title(['frame ',num2str(k)])
            k = k+1;
        end
        frame_line = k

        FS = stoploop({'(4)' 'Wait till click for', 'Turn Complete'}) ;
        while ~FS.Stop() && hasFrame(obj)
            this_frame = readFrame(obj);
            pause(ptime)
            set(imh, 'CData', this_frame);
            drawnow();
            title(['frame ',num2str(k)])
            k = k+1;
        end
        frame_finish_turn = k

        FS = stoploop({'(5)' 'Wait till click for', 'Reach chair'}) ;
        while ~FS.Stop() && hasFrame(obj)
            this_frame = readFrame(obj);
            pause(ptime)
            set(imh, 'CData', this_frame);
            drawnow();
            title(['frame ',num2str(k)])
            k = k+1;
        end
        frame_reach_chair = k
                
        FS = stoploop({'(6)' 'Wait till click for', 'Sit'}) ;
        while ~FS.Stop() && hasFrame(obj)
            this_frame = readFrame(obj);
            pause(ptime)
            set(imh, 'CData', this_frame);
            drawnow();
            title(['frame ',num2str(k)])
            k = k+1;
        end
        frame_sit = k

        time_vec = [frame_rdy_stand,frame_stand,frame_line,frame_finish_turn,frame_reach_chair,frame_sit]
%         toc
        %%% here
        csvwrite(['output/subjects/subject',num2str(subject_id), '/trial', num2str(trial_number),'/segmented_times', num2str(labeler_num), '.csv'],time_vec)
        pause
    end
end