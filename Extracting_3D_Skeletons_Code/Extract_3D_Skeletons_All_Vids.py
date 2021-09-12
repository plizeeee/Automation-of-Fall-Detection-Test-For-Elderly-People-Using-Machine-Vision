# Extracts the 3D skeleton from all videos
# Requires weights DMHS
# Also requires bounding boxes from Mask-RCNN in order to work 

import cv2
import sys
import numpy as np
import sys
caffe_root = '/dmhs/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe
import applyModelPreloadedNet_DMHS_3D
import csv
import os
import timeit




#demoDMHS - apply Deep Multitask Human Sensing (DMHS) CNN model on a single im# [pose3D, pose2D, bodyPartLabeling] = demoDMHS(img_path, output_name, multiplier_3D, multiplier_sem)
#
# OUTPUT :
# pose3D           - cell array with 6 elements (one for each stage of the
#                    DMHS) each containing the corresponding 3D pose
#                    prediction (17x3)
# pose2D           - cell array with 6 elements (one for each stage of the
#                    DMHS) each containing the corresponding 2D pose
#                    prediction (14x2)
# bodyPartLabeling - cell array with 6 elements (one for each stage of the
#                    DMHS) each containing the corresponding body part
#                    labeling mask
#
# INPUT  :
# img_path         - path to image used for testing
#                  - it should contain a single person inside a bounding box
# output_name      - output name for mat file with results corresponding to
#                    image to img_path
#                  - the mat file will be saved by default in ./data/results/
# multiplier_3D    - image scales used by the 3D pose estimation task of
#                    DMHS network
# multiplier_sem   - image scales used by the body part labeling task of
#                    DMHS network
def demoDMHS_3D(img_path, output_name, multiplier_3D):

    # Did this wrong but don't think it's important for the implementation to work
    if(len(sys.argv) == 0):
        img_path = '/data/images/im1020.jpg'
    #     img_path = './data/images/im1037.jpg';
    #     img_path = './data/images/im1054.jpg';
    #     img_path = './data/images/im2673.png';
    #     img_path = './data/images/im2788.png';
    a = len(sys.argv)
    if(len(sys.argv) < 2):
        output_name = 'results_im1020'
    #     output_name = 'results_im1037';
    #     output_name = 'results_im1054';
    #     output_name = 'results_im2673';
    #     output_name = 'results_im2788';

    if(len(sys.argv) < 3):
        multiplier_sem = np.linspace(0.4, 1, num=7)

    if(len(sys.argv) < 4):
        multiplier_3D = np.linspace(0.4, 1, num=7)

    img = cv2.imread(img_path,-1)

    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Recommended size for the X axis of img in order to obtain the best
    # network results with default multiplier_sem (0.4:0.1:1) and multiplier_3D
    # (0.7:0.1:1). For different image sizes, please revalidate multiplier_sem
    # and multiplier_3D.
    avgDimX = 386
    resizeFactor = float(avgDimX) / float(img.shape[0])
    # Dimensions seem to be reversed in Python
    img = cv2.resize(
                img, (int(resizeFactor*img.shape[1]), int(resizeFactor*img.shape[0])),
                interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('output/theoutput_im1.jpg', img)
    # switch displayModel to 1 for visualizing the results

    # Ignored these 2 lines of code, so might be an issue
    #displayMode = 1

    #addpath('./code/')

    # use_gpu = 1 for GPU usage (please note that for all 6 stages it requires
    #                            11 GB of GPU memory)
    # use_gpu = 0 for CPU usage
    # use_gpu = 1;
    # default model_id for DMHS

    ###############################                     config  note: ModelID was used to select different parameter if you wanted any (i.e  param__model1_caffe_model = 'hey', param__model1__stage = 4...                      ######
    # param = config(use_gpu, 0, model_id);
    # model = param.model(param.modelID);
    para__muse_gpu = 1
    GPUdeviceNumber = 1
    param__modelID = 1
    # Scaling parameter: starting and ending ratio of person height to image
    # height, and number of scales per octave
    # warning: setting too small starting value on non-click mode will take
    # large memory
    param__octave = 6
    caffepath = '/dmhs/caffe/python'
    print('You set your caffe in caffePath.cfg at:' + caffepath + '\n')
    sys.path.append(caffepath)

    ## Not sure how to do this via pycaffe, might be important
    # caffe.reset_all()

    if(para__muse_gpu):
        print('Setting to GPU mode, using device ID %d\n', GPUdeviceNumber)
        caffe.set_mode_gpu()
        caffe.set_device(GPUdeviceNumber)
    else:
        print('Setting to CPU mode.\n')
        caffe.set_mode_cpu()

    # Might need to be an absolute path


    param__model0__caffemodel = '/included_files/net_iter_230000.caffemodel'
    param__model0__deployFile = '/included_files/deploy.prototxt'
    param__model0__description = 'Pose 2D + Semantic Labeling + Pose3D on Human3.6M'
    param__model0__description_short = 'DMHS (H3.6M) - 6 Stages'
    param__model0__stage = 6
    param__model0__boxsize = 368
    param__model0__np = 14
    param__model0__sigma = 21
    param__model0__padValue = 128

    param__model0__limbs = np.zeros((12, 2))
    param__model0__limbs[0, 0] = 13
    param__model0__limbs[0, 1] = 14
    param__model0__limbs[1, 0] = 12
    param__model0__limbs[1, 1] = 13
    param__model0__limbs[2, 0] = 11
    param__model0__limbs[2, 1] = 12
    param__model0__limbs[3, 0] = 10
    param__model0__limbs[3, 1] = 11
    param__model0__limbs[4, 0] = 9
    param__model0__limbs[4, 1] = 13
    param__model0__limbs[5, 0] = 8
    param__model0__limbs[5, 1] = 9
    param__model0__limbs[6, 0] = 7
    param__model0__limbs[6, 1] = 8
    param__model0__limbs[7, 0] = 3
    param__model0__limbs[7, 1] = 6
    param__model0__limbs[9, 0] = 5
    param__model0__limbs[9, 1] = 6
    param__model0__limbs[10, 0] = 4
    param__model0__limbs[10, 1] = 5
    param__model0__limbs[11, 0] = 3
    param__model0__limbs[11, 1] = 2

    param__model0__part_str = ['head', 'neck', 'Rsho', 'Relb', 'Rwri',
                                   'Lsho', 'Lelb', 'Lwri',
                                   'Rhip', 'Rkne', 'Rank',
                                   'Lhip', 'Lkne', 'Lank', 'bkg']


    ######################                end of config              ##################
    # net = caffe.Net(model.deployFile, model.caffemodel, 'test');

    model_def = '/included_files/deploy.prototxt'
    model_weights = '/included_files/net_iter_230000.caffemodel'
    net = caffe.Net(model_def,  # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)

    rectangle = [1, 1, img.shape[1], img.shape[0]];
    # First output parameter of applyModelPreloadedNet_DMHS returns the cell
    # arrays containing the processed outputs of each of the network's stages.
    # Second output parameter of applyModelPreloadedNet_DMHS returns the raw
    # output of the network.


    # First output parameter of applyModelPreloadedNet_DMHS returns the cell
    # arrays containing the processed outputs of each of the network's stages.
    # Second output parameter of applyModelPreloadedNet_DMHS returns the raw
    # output of the network.
    pose3D = applyModelPreloadedNet_DMHS_3D.applyModelPreloadedNet_DMHS_3D(net, img, rectangle, multiplier_3D)
# if (~exist('./data/results/', 'dir'))
#     mkdir('./data/results/');
# end;
# save(['./data/results/' output_name '.mat'], 'pose3D', 'img');
#
# if (displayMode == 1)
#     figure;
#     % image used for testing
#     imshow(img);
#     title('Test Image');
#     figure;
#     % Estimated 3D poses corresponding to each stage (1 - 6)
#     set(gcf, 'Position', get(0,'Screensize'));
#     for i = 1 : 6
#         subplot(2, 3, i);
#         plotSkel3D(pose3D{i}, 'r');
#         title(sprintf('Pose 3D prediction - stage %d', i));
#     end
# end;


    # Refold data to put it into csv format
    pose3D_val = list()
    for stage_number in range(0,6):
        ind1 = 0
        temp = np.zeros(51)
        for ind3 in range(0,3):
            for ind2 in range(0, 17):
                temp[ind1] = pose3D[stage_number][ind2,ind3]
                ind1 = ind1+1
        pose3D_val.append(temp)

######################
    np.savetxt('output/data.csv', (pose3D_val[0], pose3D_val[1], pose3D_val[2],pose3D_val[3],pose3D_val[4],pose3D_val[5]), delimiter=',')


    return pose3D

##       Step 1: change subject number       ##
# subject_id = 9
# trial_number = 2
## Step 2 change num of frames
# number_of_frames = 995

## Step 3 ensure you downloaded 2Dpose and bboxes from TeslaK40 after running pose_est.py
start_time_global = timeit.default_timer()
# for subject_id in range(4,9+1):
#     for trial_number in (1,2,4,5):
for ind3 in range(0, 3):
    if ind3 == 0:
        subject_id = 20
        trial_number = 2
    elif ind3 == 1:
        subject_id = 25
        trial_number = 4
    else:
        subject_id = 25
        trial_number = 5
    video_folder = 'output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number) + '/'
    source_video = video_folder + 'color_output_kinect.avi'
    video_capture = cv2.VideoCapture(source_video)

    # CAP_PROP_FRAME_COUNT does not work on kinect videos for some reason. Frames are counted manually
    # number_of_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # to go to end of video
    # _, test_frame = video_capture.read()
    output_name = 'output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number) + '/image_place.jpg'

    ############ comment out line if you already ran the code once ####
    frame_path = 'output/subjects/subject'+str(subject_id) + '/trial' + str(trial_number) + '/frame'
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)

    multiplier_3D = np.zeros(4)
    multiplier_3D[0] = 0.7
    multiplier_3D[1] = 0.8
    multiplier_3D[2] = 0.9
    multiplier_3D[3] = 1.0

    # img_path = 'output/frame_im4.jpg'
    #x = demoDMHS_3D(img_path, output_name, multiplier_3D)
    bboxes = np.loadtxt('output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number) + '/bboxes' +'.csv' , delimiter=',')
    number_of_frames = bboxes.shape[1]
    outputs = list()

    # For debugging and checking every 10 frames (filename starts with 1 for MATLAB convention
    f = 1
    pose3D_val = list()
    start_time = timeit.default_timer()

    length = int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    # Don't use this specific loop in the end, since it quantizes number of outputs.
    # Or set val2 range to 1 and val 1 range to number_of_frames/1
    for val1 in range(0, int((number_of_frames))):
        aa = 4
        for val2 in range(0, 1):    # This inner loop was used for debugging, to save the values after n iterations
            success, frame = video_capture.read()
            # zzz = 'output/frame/frame'+ str(f) + '.jpg'
            frame_path = 'output/subjects/subject'+str(subject_id) + '/trial' + str(trial_number) +'/frame/frame'+ str(f) + '.jpg'
            cropped_img_path = 'output/subjects/subject'+str(subject_id) + '/trial' + str(trial_number) +'/frame/cropped_im'+ str(f) + '.jpg'
            cv2.imwrite(frame_path,frame)
            # immm = cv2.imread(frame_path)
            xmin = max(int(round(bboxes[0][f-1]- 10)),0)
            ymin = max(int(round(bboxes[1][f-1]- 10)),0)
            xmax = min(int(round(bboxes[2][f-1] + 10)),frame.shape[1])
            ymax = min(int(round(bboxes[3][f-1] + 10)),frame.shape[0])
            cropped_im = np.zeros((ymax-ymin,xmax-xmin,3))
            y = ymin
            for i in range(0,ymax-ymin):
                x = xmin
                for j in range(0, xmax - xmin):
                    for k in range(0,3):
                        cropped_im[i,j,k] = frame[y,x,k]
                    x = x+1
                y = y+1
            cv2.imwrite(cropped_img_path, cropped_im)
            outputs.append(demoDMHS_3D(cropped_img_path, output_name, multiplier_3D))
            print('Subject '+ str(subject_id) + ', trial ' + str(trial_number))
            print('Frame ' + str(val1), ' of ' + str(number_of_frames))
            # Refold data to put it into csv format
            # for stage_number in range(0, 6):
            ind1 = 0
            temp = np.zeros(51)
            for ind3 in range(0, 3):
                for ind2 in range(0, 17):
                    temp[ind1] = outputs[f-1][5][0:17][ind2][ind3]
                    ind1 = ind1 + 1
            pose3D_val.append(temp)
            f = f+1
        # Save the 3d pose of the last 20 frames as a csv file
        # np.savetxt('output/data2.csv',
        #                (pose3D_val[g], pose3D_val[g+1], pose3D_val[g+2], pose3D_val[g+3], pose3D_val[g+4], pose3D_val[g+5],pose3D_val[g+6],pose3D_val[g+7],pose3D_val[g+8],pose3D_val[g+9],pose3D_val[g+10],pose3D_val[g+11],pose3D_val[g+12],pose3D_val[g+13],pose3D_val[g+14],pose3D_val[g+15],pose3D_val[g+16],pose3D_val[g+17],pose3D_val[g+18],pose3D_val[g+19]),
        #                delimiter=',')
        np.savetxt('output/subjects/subject' + str(subject_id) + '/trial' + str(trial_number) + '/rgb_skeleton.csv',
                   (pose3D_val),
                   delimiter=',')

    # np.savetxt('output/data4.csv', (pose3D_val[0], pose3D_val[1], pose3D_val[2],pose3D_val[3],pose3D_val[4],pose3D_val[5]), delimiter=',')


    # for f in range(1, number_of_frames):
    #     success, frame = video_capture.read()
    #     cv2.imwrite('output/frame.jpg', frame)
    #     xmin = int(round(bboxes[0][f]- 10))
    #     ymin = int(round(bboxes[1][f]- 10))
    #     xmax = int(round(bboxes[2][f] + 10))
    #     ymax = int(round(bboxes[3][f] + 10))
    #     cropped_im = np.zeros((ymax-ymin,xmax-xmin,3))
    #     y = ymin
    #     for i in range(0,ymax-ymin):
    #         x = xmin
    #         for j in range(0, xmax - xmin):
    #             for k in range(0,3):
    #                 cropped_im[i,j,k] = frame[y,x,k]
    #             x = x+1
    #         y = y+1
    #     cv2.imwrite('output/cropped_im.jpg', cropped_im)
    #     outputs.append(demoDMHS_3D(img_path, output_name, multiplier_3D))
    # Print that the video is finished being processed
    # Also print the amount of time that was used to process the video
    # It can take a very long time to process each video, since the algorithm needed to be applied
    # to the videos on a frame-by-frame basis.
    #
    print('-----------------------Video finished----------------------')
    print(timeit.default_timer()-start_time)
print(timeit.default_timer()-start_time_global)
