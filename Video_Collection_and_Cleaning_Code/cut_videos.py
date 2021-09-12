# This code cuts the videos to the timing of when the participants started the tests
# Change the subject ID and trial number for each video in the dataset
#
import os
import shutil
import cv2
import numpy as np
## parameters ##
subject_id = 20

start_frame = 230.0
end_frame = 520.0
trial_number = 2
frame_rate = 28.3   # Approximately Kinect's frame rate

## Note: I could not perfectly allign the camera streams with the rgb camera stream,
# Since the kinect video properties do not work properly.

start_time = start_frame/frame_rate
end_time = end_frame/frame_rate

time_dif = end_time - start_time

if not os.path.isfile('output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number) + '/cut_times.csv'):
    if trial_number ==1:
        # make copy of original videos in case they need to be used later
        shutil.copytree('output/subjects/subject' + str(subject_id),
                        'output/subjects/subject' + str(subject_id) + 'temp'
                        )

        os.rename('output/subjects/subject' + str(subject_id),
                  'output/subjects/subject' + str(subject_id) + '_uncut'
                  )

        os.rename('output/subjects/subject' + str(subject_id) + 'temp',
                  'output/subjects/subject' + str(subject_id)
                  )



    # os.system('ffmpeg -i output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number)+'/color_output_kinect.avi -vcodec h264 -ss 00:00:' + str(start_time) + ' -t 00:00:' + str(time_dif) + 'output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number) + '/temp_name_k.avi')


    output_vid = cv2.VideoWriter('output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number) + '/temp_name_k.avi',
                                      cv2.VideoWriter_fourcc(*'XVID'),
                                      frame_rate,
                                      (1920, 1080))

    video_capture = cv2.VideoCapture('output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number)+'/color_output_kinect.avi')

    success = True
    ind =1
    while success:
        success, input_frame = video_capture.read()
        if ind>=start_frame and ind<=end_frame:
            output_vid.write(input_frame)
        ind = ind + 1
        print(ind)
    video_capture.release()
    output_vid.release()

    os.remove('output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number)+'/color_output_kinect.avi')
    os.rename('output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number) + '/temp_name_k.avi',
              'output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number)+'/color_output_kinect.avi'
              )


    for i in range(1,6):
        os.system('ffmpeg -i output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number)+'/camera_' + str(i) + '.avi -vcodec h264 -ss 00:00:' + str(start_time) + ' -t 00:00:' + str(time_dif) + ' output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number) + '/temp_name' + str(i) + '.avi')
        os.remove('output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number)+ '/camera_' + str(i) + '.avi')
        os.rename('output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number) + '/temp_name' + str(i) + '.avi',
                  'output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number) + '/camera_' + str(i) + '.avi'
                  )
    # Save cut times
    # This is to ensure I replicate my processing steps
    np.savetxt('output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number) + '/cut_times.csv',
                       ([start_frame,end_frame]),
                       delimiter=',')
# If video is already trimmed print an error message
else:
    print('Error videos already trimmed')