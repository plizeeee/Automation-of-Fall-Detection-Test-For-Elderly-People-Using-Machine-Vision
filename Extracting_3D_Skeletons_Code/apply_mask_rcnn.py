import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging

import timeit
import sys
sys.path.append('/usr/local/caffe2_build')

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.core.rpn_generator as rpn_engine
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils
import detectron.datasets.dummy_datasets as dummy_datasets
import pycocotools.mask as mask_util
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
plt.interactive(False)
import keyptsfile
from PIL import Image
from detectron.utils.collections import AttrDict


################################Follow steps below

##### Step1 : Set subject number ##########
# subject_id = 9
# trial_number = 2


##########################
# setup caffe2
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

model_file_location = '/included_files/weights_and_models/kp_only_model.yaml'
weight_file_location = '/included_files/weights_and_models/kp_only_weights.pkl'

# model_file_location = '/included_files/weights_and_models/mask_model.yaml'
# weight_file_location = '/included_files/weights_and_models/mask_weights.pkl'

# setup detectron
logger = logging.getLogger(__name__)
merge_cfg_from_file(model_file_location)
cfg.NUM_GPUS = 1
assert_and_infer_cfg(cache_urls=False)
model = infer_engine.initialize_model_from_cfg(weight_file_location)

# 1 to 8. ***Note: comment out this loop if you just want to do 1 video at a time
# for subject_id in range(4,9+1):
for ind3 in range(0,3):
    if ind3 == 0:
        subject_id = 20
        trial_number = 2
    elif ind3 == 1:
        subject_id = 25
        trial_number = 4
    else:
        subject_id = 25
        trial_number = 5

    start_time_vid = timeit.default_timer()
    # for trial_number in (1,2,4,5):

    ##### Step2 : Set number of frames (or maybe I can get it automatiaclly) ##########
    # number_of_frames = 996
    ##### Step3 upload data to K40 prior to processing ########

    ###### Step4 (after running code) download results from TeslaK40 ########
    # Results include 2D pose and bounding boxes in the subject/subject# folder

    # setup video
    # source_folder = '/included_files/'
    # video_folder = source_folder + 'video_files/'
    # output_dir = '/output/'
    # source_video = video_folder + 'colour_output' + str(subject_id)+ '.avi'

    ####### Add these lines to the TeslaK40 prior to downloading files from the device
    # os.mkdir('output/subjects/subject'+str(subject_id))
    # os.mkdir('output/subjects/subject'+str(subject_id)+'source_video')
    # os.mkdir('output/subjects/subject'+str(subject_id)+'frames')  #Frames of video for convenience

    source_video = 'output/subjects/subject'+str(subject_id) + '/trial' + str(trial_number) + '/color_output_kinect' + '.avi'
    video_capture = cv2.VideoCapture(source_video)

    # number_of_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # to go to end of video

    frames_per_second = 28.0
    # vv, test_frame = video_capture.read()
    #
    # if (vv):
    #     print(3)
    # height = test_frame.shape[0]
    # width = test_frame.shape[1]


    ind = 0; # Person #0 in image
    # start annotating
    # numframes = 10

    # ycoord_noseall = np.zeros(number_of_frames)
    # xcoord_noseall = np.zeros(number_of_frames)
    # time = np.zeros(number_of_frames)
    video_keypoints = list()
    bboxes = list()

    # bboxesaa = [1,2,3]
    # print('/output/subjects/subject'+str(subject_id)+ '2dpose' + str(subject_id)+'.csv')
    # np.savetxt('/output/subjects/subject'+str(subject_id)+ '2dpose' + str(subject_id)+'.csv',
    #            (bboxesaa), delimiter=',')
    #
    # np.savetxt('/output/subjects/subject'+str(subject_id)+'bboxes'+str(subject_id)+'.csv',
    #            (bboxesaa), delimiter=',')

    # for f in range(0, number_of_frames):
    success = True
    number_of_frames = 0
    while success:
        success, frame = video_capture.read()
        if success:

            start_time = timeit.default_timer()
            with c2_utils.NamedCudaScope(0):
                cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                    model, frame, None
                )
                time_taken = timeit.default_timer() - start_time
                print(str(time_taken))
            annotated_frame = vis_utils.vis_one_image_opencv(
                frame,  # BGR -> RGB for visualization
                cls_boxes,
                cls_segms,
                cls_keyps,
                dataset=dummy_datasets.get_coco_dataset(),
                show_box=True,
                show_class=True
            )
            # index that is most likely associate with the person in the video
            # If there are multiple non-occluded people in the video this will
            # need to be done with tracking.
            person_ind = np.argmax(cls_boxes[1][:,4])
            bboxes.append(cls_boxes[1][person_ind,0:-1])
            # xcoord_noseall[ind] = cls_keyps[1][person_ind][0,1]
            # ycoord_noseall[ind] = cls_keyps[1][person_ind][1,1]
            # time[ind] = (float(ind)/20); # Time in video
            # ind = ind+1
            #     # example for extracting the nose and the hips (can use
            # # basic interpolation to find mid point)
            # xcoord_nose = cls_keyps[1][person_ind][0,1]
            # ycoord_nose = cls_keyps[1][person_ind][1,1]
            #
            # xcoord_lefthip = cls_keyps[1][person_ind][0,11]
            # ycoord_lefthip = cls_keyps[1][person_ind][1,11]
            #
            # xcoord_righthip = cls_keyps[1][person_ind][0,12]
            # ycoord_righthip = cls_keyps[1][person_ind][1,12]
            #
            # midhipx =  (xcoord_lefthip+xcoord_righthip)/2
            # midhipy = (ycoord_lefthip + ycoord_righthip)/2
            video_keypoints.append(keyptsfile.HumanKeypoints(cls_boxes, cls_keyps))
            number_of_frames = number_of_frames+1
            # Masks of person
        # masks = mask_util.decode(cls_segms[1])

        # plt.subplot(211)
        # plt.imshow(masks[:,:,ind])
        # plt.subplot(212)
        # plt.imshow(masks[:,:,ind], cmap='Greys', interpolation='nearest')
        # plt.savefig('output/' + 'output.jpg')
        #
        # plt.subplot(211)
        # plt.imshow(masks[:, :, ind+1])
        # plt.subplot(212)
        # plt.imshow(masks[:, :, ind+1], cmap='Greys', interpolation='nearest')
        # plt.savefig('output/' + 'output2.jpg')
        #
        # plt.subplot(211)
        # plt.imshow(frame)
        # plt.subplot(212)
        # plt.imshow(frame, cmap='Greys', interpolation='nearest')
        # plt.savefig('output/' + 'output3.jpg')
        #
        # plt.show()


# localmins = np.r_[True, ycoord_noseall[1:] < ycoord_noseall[:-1]] & np.r_[ycoord_noseall[:-1] < ycoord_noseall[1:], True]
# inds = ycoord_noseall[localmins]
# # clean up

    # j = 0;
    # min_ind = np.zeros(40)
    # tot = 0
    # for i in range(5,194):
    #     # Must be a substantial drop
    #     ll = 0;
    #     for k in range(i-5, i+6):
    #         tot = ycoord_noseall[k] + tot
    #         ll = ll+1
    #     mean_val = tot/ll
    #     if (ycoord_noseall[i]< ycoord_noseall[i-1]) and ycoord_noseall[i]< ycoord_noseall[i+1] and ycoord_noseall[i] < 0.9*mean_val:
    #         min_ind[j] = i
    #         j = j + 1
    nose_x = np.zeros(number_of_frames)
    left_hip_x = np.zeros(number_of_frames)
    right_hip_x = np.zeros(number_of_frames)
    left_wrist_x = np.zeros(number_of_frames)
    right_wrist_x = np.zeros(number_of_frames)
    left_shoulder_x = np.zeros(number_of_frames)
    right_shoulder_x = np.zeros(number_of_frames)
    left_knee_x = np.zeros(number_of_frames)
    right_knee_x = np.zeros(number_of_frames)
    left_ankle_x = np.zeros(number_of_frames)
    right_ankle_x = np.zeros(number_of_frames)
    nose_y = np.zeros(number_of_frames)
    left_hip_y = np.zeros(number_of_frames)
    right_hip_y = np.zeros(number_of_frames)
    left_wrist_y = np.zeros(number_of_frames)
    right_wrist_y = np.zeros(number_of_frames)
    left_shoulder_y = np.zeros(number_of_frames)
    right_shoulder_y = np.zeros(number_of_frames)
    left_knee_y = np.zeros(number_of_frames)
    right_knee_y = np.zeros(number_of_frames)
    left_ankle_y = np.zeros(number_of_frames)
    right_ankle_y = np.zeros(number_of_frames)

    bbox1 = np.zeros(number_of_frames)
    bbox2 = np.zeros(number_of_frames)
    bbox3 = np.zeros(number_of_frames)
    bbox4 = np.zeros(number_of_frames)



    for i in range(0,number_of_frames):
        nose_x[i] = video_keypoints[i].nose[0]
        left_hip_x[i] = video_keypoints[i].left_hip[0]
        right_hip_x[i] = video_keypoints[i].right_hip[0]
        left_wrist_x[i] = video_keypoints[i].left_wrist[0]
        right_wrist_x[i] = video_keypoints[i].right_wrist[0]
        left_shoulder_x[i] = video_keypoints[i].left_shoulder[0]
        right_shoulder_x[i] = video_keypoints[i].right_shoulder[0]
        left_knee_x[i] = video_keypoints[i].left_knee[0]
        right_knee_x[i] = video_keypoints[i].right_knee[0]
        left_ankle_x[i] = video_keypoints[i].left_ankle[0]
        right_ankle_x[i] = video_keypoints[i].right_ankle[0]
        nose_y[i] = video_keypoints[i].nose[1]
        left_hip_y[i] = video_keypoints[i].left_hip[1]
        right_hip_y[i] = video_keypoints[i].right_hip[1]
        left_wrist_y[i] = video_keypoints[i].left_wrist[1]
        right_wrist_y[i] = video_keypoints[i].right_wrist[1]
        left_shoulder_y[i] = video_keypoints[i].left_shoulder[1]
        right_shoulder_y[i] = video_keypoints[i].right_shoulder[1]
        left_knee_y[i] = video_keypoints[i].left_knee[1]
        right_knee_y[i] = video_keypoints[i].right_knee[1]
        left_ankle_y[i] = video_keypoints[i].left_ankle[1]
        right_ankle_y[i] = video_keypoints[i].right_ankle[1]
        bbox1[i] =bboxes[i][0]
        bbox2[i] = bboxes[i][1]
        bbox3[i] = bboxes[i][2]
        bbox4[i] = bboxes[i][3]



# keyptsfile.save_keypoints_to_json(video_keypoints,'output/subjects/subject'+str(subject_id) + '/', 'keypts_json')
# np.savetxt('output/2dpose'+ str(subject_id)+'.csv',
#            (nose_x,nose_y,left_hip_x,left_hip_y,right_hip_x,right_hip_y,left_wrist_x,left_wrist_y,right_wrist_x,right_wrist_y,left_shoulder_x,left_shoulder_y,right_shoulder_x,right_shoulder_y,left_knee_x,left_knee_y,right_knee_x,right_knee_y,left_ankle_x,left_ankle_y,right_ankle_x,right_ankle_y), delimiter=',')


    np.savetxt('/output/subjects/subject'+str(subject_id) + '/trial' + str(trial_number) + '/2dpose' + '.csv',
                   (nose_x,nose_y,left_hip_x,left_hip_y,right_hip_x,right_hip_y,left_wrist_x,left_wrist_y,right_wrist_x,right_wrist_y,left_shoulder_x,left_shoulder_y,right_shoulder_x,right_shoulder_y,left_knee_x,left_knee_y,right_knee_x,right_knee_y,left_ankle_x,left_ankle_y,right_ankle_x,right_ankle_y), delimiter=',')

    np.savetxt('/output/subjects/subject'+str(subject_id) + '/trial' + str(trial_number) + '/bboxes'+'.csv',
                   (bbox1,bbox2,bbox3,bbox4), delimiter=',')

    # print(min_ind)
    # plt.plot(time,ycoord_noseall)
    # plt.savefig('/output/' + 'output_pose1.jpg')
    # plt.plot(time,xcoord_noseall)
    # plt.savefig('/output/' + 'output_pose2.jpg')
    # print(midhipy)
    # plt.show()
    # plt.savefig('/output/' + 'output_pose.jpg')
    # print(midhipy)
    # print(localmins)
    # print(inds)
    video_capture.release()
    cv2.destroyAllWindows()
    print('-------------------------- Finished a video -----------')
    print('time_taken = ' + str(timeit.default_timer()-start_time_vid))