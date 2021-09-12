# function predictions = applyModelPreloadedNet_DMHS_3D(net, oriImg, param, rectangle, multiplier_3D)
import cv2
import sys
import numpy as np
import os
import sys
caffe_root = '/dmhs/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe

## Set multipler 3D to [0.7,0.8,0.9,1] for images of size 368 by 368

## The following is what needs to be input as the net parameter of the applyModelpreloadedNet_3d function###
# caffe.set_mode_gpu()
#
# model_def = '/included_files/deploy.prototxt'
# model_weights = '/included_files/net_iter_230000.caffemodel'
#
# net = caffe.Net(model_def,      # defines the structure of the model
#                 model_weights,  # contains the trained weights
#                 caffe.TEST)     # use test mode (e.g., don't perform dropout)
##                                                                              ##
def applyModelPreloadedNet_DMHS_3D(net, oriImg, rectangle, multiplier_3D):
    # Select model and other parameters from param
    #model = param.model(param.modelID);
   ###### part of config function giving model #####
    # Might need to be an absolute path


    boxsize = 368
    np_val = 14
    nstage = 6
    sigma_val = 21
    padValue = 128
    #predictions = cell(1, 1);
    predictions = list()

    # Apply multitask network model
    # set the center and roughly scale range(overwrite the config!) according to the rectangle
    x_start = max(rectangle[0], 1)
    x_end = min(rectangle[0] + rectangle[2],oriImg.shape[1])
    y_start = max(rectangle[1], 1)
    y_end = min(rectangle[1] + rectangle[3], oriImg.shape[0])
    center = [(x_start + x_end) / float(2), (y_start + y_end) / float(2)]

    # buffer_score = cell(nstage, 3, length(multiplier_3D));
    # score_3D = cell(nstage, length(multiplier_3D));
    # pad = cell(1, length(multiplier_3D));
    # ori_size = cell(1, length(multiplier_3D));

    # Initialize cell array-like data structures in python
    buffer_score = list()
    for i in range(0, nstage):
        buffer_score.append(list())
        for j in range(0, 3):   # Pretty sure index 2 and 3 are for 2D pose and semantic segmentation channels respectively, but keeping it anyway just in case
            buffer_score[i].append(list())
            for k in range(0, len(multiplier_3D)):
                buffer_score[i][j].append(list())
    # buffer_score[2][2][5].append('whaat')
    # buffer_score[2][2][4].append(1.2)

    score_3D = list()
    for i in range(0, nstage):
        score_3D.append(list())
        for j in range(0, len(multiplier_3D)):
            score_3D[i].append(list())

    # score_3D[2][3].append('whaat')

    pad = list()
    for i in range(0, 1):
        pad.append(list())
        for j in range(0, len(multiplier_3D)):
            pad[i].append(list())

    ori_size = list()
    for i in range(0, 1):
        ori_size.append(list())
        for j in range(0, len(multiplier_3D)):
            ori_size[i].append(list())

    # net = caffe.Net(model.deployFile, model.caffemodel, 'test');
    # change outputs to enable visualizing stagewise results
    # note this is why we keep out own copy of m - files of caffe wrapper

    for m in range(0,len(multiplier_3D)):
        scale = multiplier_3D[m]
        # imageToTest = imresize(oriImg, scale)
        imageToTest = cv2.resize(
            oriImg, (int(scale * oriImg.shape[1]), int(scale * oriImg.shape[0])),
            interpolation=cv2.INTER_CUBIC)
        ori_size[0][m] = imageToTest.shape
        cv2.imwrite('output/theoutput_im2.jpg', imageToTest)
        center_s = list()
        center_s.append(center[0] * scale)
        center_s.append(center[1] * scale)


        ## model.padValue = 128
        padvals = padAround(imageToTest, boxsize, center_s, padValue)
        imageToTest = padvals[0]
        pad[0][m] = padvals[1]

        cv2.imwrite('output/theoutput_im3.jpg', imageToTest)

        imageToTest = preprocess(imageToTest, 0.5, boxsize, sigma_val)

        img_out_temp = np.zeros((boxsize,boxsize,3),np.uint8)
        for i in range(0, boxsize):
            for j in range(0, boxsize):
                for k in range(0, 3):
                    img_out_temp[i, j, k] = np.uint8((imageToTest[k, i, j]+0.5)*256)

        cv2.imwrite('output/theoutput_im4.jpg', img_out_temp)

        nnresults = applyDNN(imageToTest, net)

        for j in range(0,6):
            buffer_score[j][1][m] = nnresults[j]

        # if (~sum(find(multiplier_3D == scale)) == 0)
        # idx = find(multiplier_3D == scale)
        # score_3D(:, idx) = buffer_score(:, m)
        # Find if the multiplier is equal the scale (which it must be according to the variables declared
        # in this function lol. But they had this so just making sure
        for index_mul in range(0,len(multiplier_3D)):
            if scale==multiplier_3D[index_mul]:
                idx = index_mul
                for j in range(0,6):
                    score_3D[j][idx]= buffer_score[j][1][m]


# Process 3D POSE
    predictions = list()
    for s in range(0,nstage):
        est = np.zeros((17, 3))
        for m in range(0,len(multiplier_3D)):
            ##
            temp = score_3D[s][m]

            # est = est + reshape(temp, [17 3])
            z_ind = 0
            temp_reshaped = np.zeros((17,3))
            for ind2 in range(0,3):
                for ind1 in range(0,17):
                    temp_reshaped[ind1,ind2] = temp[0,z_ind]
                    z_ind = z_ind+1
            est = est + temp_reshaped

        # predictions{s} = est / length(multiplier_3D)
        predictions.append(est / len(multiplier_3D))
    return predictions

# function img_out = preprocess(img, mean, param)
def preprocess(img, mean,boxsize,sigma):
    # Note: Floats in Python are double precision by default
    img_out_temp = np.float64(img) / np.float64(256.00)
    img_out_temp= np.float64(img_out_temp) - mean;
    # img_out = permute(img_out, [2 1 3]);
    ###
    img_out_temp = np.transpose(img_out_temp, (1, 0, 2))

    # Pretty sure open cv reads the image as RGB, so no need for this step
    # img_out = img_out(:,:, [3 2 1]); # BGR for opencv training in caffe !!!!!

    # No need for this line since I explicitly mention the boxsize as the input now
    # boxsize = param.model(param.modelID).boxsize;

    ##
    img_out = np.zeros((4,boxsize,boxsize))

    ##
    # Indices 0 to 2 copying image into im_out
    for i in range(0,boxsize):
        for j in range(0, boxsize):
            for k in range(0,3):
                img_out[k,i,j] = img_out_temp[j,i,k]

    centerMapCell = produceCenterLabelMap([boxsize,boxsize], boxsize / 2, boxsize / 2, sigma)

    # Index 3, copy gaussian peak into im_out
    for i in range(0,boxsize):
        for j in range(0, boxsize):
            img_out[3,i,j] = centerMapCell[i,j]

    return img_out

# function scores = applyDNN(images, net)
def applyDNN(images, net):
    # input_data = {single(images)};
    # do forward pass to get scores (Input should be single precision float in MATLAB)
    # scores are now Width x Height x Channels x Num
    # net.forward(input_data);
    net.blobs['data'].data[...] = images
    output = net.forward()

    # scores = cell(6, 1);
    scores = list()

    # for s = 1:6
    #     string_to_search_v1 = sprintf('pose_3D_pose_regress_%d', s);
    #
    #     blob_id = net.name2blob_index(string_to_search_v1);
    #     scores{s} = net.blob_vec(blob_id).get_data();
    # net.forward(input_data);
    # 6+1 to remember that range stops at 6
    for i in range(1,6+1):
        scores.append(output[('pose_3D_pose_regress_'+str(i))])

    return scores

# function[img_padded, pad] = padAround(img, boxsize, center, padValue)
def padAround(img, boxsize, center, padValue):
    center = [round(center[0]), round(center[1])]
    h = img.shape[0]
    w = img.shape[1]
    pad = np.zeros(4)
    pad[0] = boxsize / 2 - center[1]  # up
    pad[2] = boxsize / 2 - (h - center[1])  # down
    pad[1] = boxsize / 2 - center[0]  # left
    pad[3] = boxsize / 2 - (w - center[0])  # right
    ##
    # pad_up = repmat(img(1,:,:), [pad(1) 1 1])*0 + padValue
    # img_padded = [pad_up;img]
    #
    # pad_left = repmat(img_padded(:, 1,:), [1 pad(2) 1])*0 + padValue
    # img_padded = [pad_left img_padded]
    #
    # pad_down = repmat(img_padded(end,:,:), [pad(3) 1 1])*0 + padValue
    # img_padded = [img_padded;pad_down]
    #
    # pad_right = repmat(img_padded(:, end,:), [1 pad(4) 1])*0 + padValue
    # img_padded = [img_padded pad_right]

    # in case all if statements not met set img_padded to the image
    img_padded = img
    if pad[0] > 0:
        # pad_up = repmat(img(1,:,:), [pad(1) 1 1])*0 + padValue;
        # img_padded = [pad_up;img];

        # pad_up = np.tile(img[0, :, :], (int(pad[0]), 1, 1)) * 0 + padValue
        pad_up = np.tile(np.zeros((1, img.shape[1], 3), np.uint8), (int(pad[0]), 1, 1)) * 0 + padValue
        img_padded = np.concatenate((pad_up, img), axis=0)

    if pad[1] > 0:
        # pad_left = repmat(img_padded(:, 1,:), [1 pad(2) 1])*0 + padValue;
        # img_padded = [pad_left img_padded];

        # a = (img_padded[:, 0, :])
        # b = np.tile(img_padded[:, 0, :], (1, int(pad[1]), 1))
        # c = (1, int(pad[1]), 1)
        # d = np.tile(img_padded[:, 0, :], (1, int(pad[1]), 1)) * 0
        # # pad_left = np.tile(img_padded[:,0,:], (1,int(pad[1]),1)) * 0 + padValue
        # f = np.zeros((img_padded.shape[0],1,3))
        pad_left = np.tile(np.zeros((img_padded.shape[0], 1, 3), np.uint8), (1, int(pad[1]), 1)) * 0 + padValue
        img_padded = np.concatenate((pad_left, img_padded), axis=1)

    if pad[2] > 0:
        # pad_down = repmat(img_padded(end,:,:), [pad(3) 1 1])*0 + padValue;
        # img_padded = [img_padded;pad_down]

        last_ind1 = img.shape[0] - 1
        # pad_down = np.tile(img[last_ind1, :, :], (int(pad[2]), 1, 1)) * 0 + padValue
        pad_down = np.tile(np.zeros((1, img_padded.shape[1], 3), np.uint8), (int(pad[2]), 1, 1)) * 0 + padValue
        img_padded = np.concatenate((img_padded, pad_down), axis=0)
    if pad[3] > 0:
        # pad_right = repmat(img_padded(:, end,:), [1 pad(4) 1])*0 + padValue;
        # img_padded = [img_padded pad_right];

        last_ind2 = img.shape[1] - 1
        pad_right = np.tile(np.zeros((img_padded.shape[0], 1, 3), np.uint8), (1, int(pad[3]), 1)) * 0 + padValue
        img_padded = np.concatenate((img_padded, pad_right), axis=1)

    center[0] = round(center[0] + max(0, pad[1]))
    center[1] = round(center[1] + max(0, pad[0]))

    # img_padded = img_padded(center(2) - (boxsize / 2 - 1):center(2) + boxsize / 2, center(1) - (boxsize / 2 - 1): center(
    #         1) + boxsize / 2,:); # cropping if needed
    aaa = [int(center[1] - boxsize / 2), int(center[1] + boxsize / 2 - 1), int(center[0] - boxsize / 2),
           int(center[0] + boxsize / 2 - 1), 3]
    img_padded = img_padded[int(round(center[1] - boxsize / 2)):int(round(center[1] + boxsize / 2)),
                 int(round(center[0] - boxsize / 2)): int(round(center[0] + boxsize / 2)), :]  # cropping if needed

    return [img_padded, pad]

# function label = produceCenterLabelMap(im_size, x, y, sigma)
def produceCenterLabelMap(im_size, x, y, sigma):
    # this function generates a gaussian peak centered at position(x, y)
    # it is only for center map in testing
    #     [X, Y] = meshgrid(1:im_size(1), 1: im_size(2))
    #     X = X - x;
    #     Y = Y - y;
    #     D2 = X. ^ 2 + Y. ^ 2;
    #     Exponent = D2. / 2.0. / sigma. / sigma;
    #     label{1} = exp(-Exponent);
    label = np.zeros((im_size[0],im_size[1]))
    for X in range (1,im_size[0]+1):
        for Y in range(1, im_size[1]+1):
            label[X-1,Y-1] = np.exp(-((X-x)**2+(Y-y)**2)/(2.0*sigma*sigma))
    return label