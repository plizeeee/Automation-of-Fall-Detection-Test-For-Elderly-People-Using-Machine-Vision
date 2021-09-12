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

from detectron.utils.collections import AttrDict

# setup caffe2
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

model_file_location = '/included_files/weights_and_models/kp_only_model.yaml'
weight_file_location = '/included_files/weights_and_models/kp_only_weights.pkl'

model_file_location = '/included_files/weights_and_models/mask_model.yaml'
weight_file_location = '/included_files/weights_and_models/mask_weights.pkl'


# setup detectron
logger = logging.getLogger(__name__)
merge_cfg_from_file(model_file_location)
cfg.NUM_GPUS = 1
assert_and_infer_cfg(cache_urls=False)
model = infer_engine.initialize_model_from_cfg(weight_file_location)

# setup video
source_folder = '/included_files/'
video_folder = source_folder + 'video_files/'
output_dir = '/output/'
source_video = video_folder + 'TUG_Test_camera_1.avi'
video_capture = cv2.VideoCapture(source_video)

number_of_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # to go to end of video
# number_of_frames = 600

frames_per_second = 20.0
_, test_frame = video_capture.read()
height = test_frame.shape[0]
width = test_frame.shape[1]

# Create VideoWriter object
annotated_video = cv2.VideoWriter(output_dir + 'output.avi', cv2.VideoWriter_fourcc(*'XVID'),
                                  frames_per_second,
                                  (width, height))

# start annotating
for f in range(1, number_of_frames):
    success, frame = video_capture.read()
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
    annotated_video.write(annotated_frame)

video_capture.release()
annotated_video.release()
cv2.destroyAllWindows()
