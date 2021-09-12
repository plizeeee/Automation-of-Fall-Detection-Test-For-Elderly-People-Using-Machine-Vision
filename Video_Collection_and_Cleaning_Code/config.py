function param = config(use_gpu, device_number, modelID)

def config(use_gpu, device_number, modelID):
    # CPU mode (0) or GPU mode (1)
    # friendly warning: CPU mode may take a while
    param.use_gpu = use_gpu

    # GPU device number (doesn't matter for CPU mode)
    GPUdeviceNumber = device_number

    # Select model (default: 1)
    # 1: 'Multitask Human3.6M - 6 Stages'

    # Always set modelID to what the use picks (go with 1. Not sure why, just go with it)
    # if(nargin < 3)
    #     param.modelID = 1
    # else
    param.modelID = modelID

    # Scaling parameter: starting and ending ratio of person height to image
    # height, and number of scales per octave
    # warning: setting too small starting value on non-click mode will take
    # large memory
    param.octave = 6


    # WARNING! Adjust the path to your caffe accordingly!
    caffepath = '/dmhs/caffe/python'

    print('You set your caffe in caffePath.cfg at:' + caffepath + '\n')

    ##
    addpath(caffepath);
    caffe.reset_all();
    if(param.use_gpu)
        fprintf('Setting to GPU mode, using device ID %d\n', GPUdeviceNumber);
        caffe.set_mode_gpu();
        caffe.set_device(GPUdeviceNumber);
    else
        fprintf('Setting to CPU mode.\n');
        caffe.set_mode_cpu();
    end

    # Might need to be an absolute path
param.model[0].caffemodel = '/included_files/net_iter_230000.caffemodel';
param.model[0].deployFile = '/included_files/deploy.prototxt';

param.model[0].description = 'Pose 2D + Semantic Labeling + Pose3D on Human3.6M';
param.model[0].description_short = 'DMHS (H3.6M) - 6 Stages';
param.model[0].stage = 6;
param.model[0].boxsize = 368;
param.model[0].np = 14;
param.model[0].sigma = 21;
param.model[0].padValue = 128;
param.model[0].limbs = [13 14; 12 13; 11 12; 10 11; 9 13; 8 9; 7 8; 3 6; 5 6; 4 5; 3 2; 2 1];

param__model0__limbs = list()

for i in range(0,12):
    param__model0__limbs.append(list())
param__model0__limbs[0][0] = 13
param__model0__limbs[0][1] = 14
param__model0__limbs[1][0] = 12
param__model0__limbs[1][1] = 13






param.model[0].part_str = ['head', 'neck', 'Rsho', 'Relb', 'Rwri',
                             'Lsho', 'Lelb', 'Lwri',
                             'Rhip', 'Rkne', 'Rank',
    'Lhip', 'Lkne', 'Lank', 'bkg'];