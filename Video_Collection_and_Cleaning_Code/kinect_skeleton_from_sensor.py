import timeit
import sys
from openni import openni2, nite2
import time
import cv2
import numpy as np

record_time = 9000 # seconds

openni2.initialize('/usr/lib')
nite2.initialize('/home/jcamero4/src/NiTE-2.0.0/Redist')

userTracker = nite2.UserTracker.open_any()

n = 1/float(29)
start_time = timeit.default_timer()

# Item names
items = list()

while timeit.default_timer() - start_time < record_time:
    start_time_frame = timeit.default_timer()
    frame = userTracker.read_frame()

    if frame.users:
        for user in frame.users:
            if user.is_new():
                print("New human detected! Calibrating...")
                userTracker.start_skeleton_tracking(user.id)
            elif user.skeleton.state == nite2.SkeletonState.NITE_SKELETON_TRACKED:
                torso = user.skeleton.joints[nite2.JointType.NITE_JOINT_TORSO] # 1 No pelvis like DMHS, think torso might mean the same thing
                right_hip = user.skeleton.joints[nite2.JointType.NITE_JOINT_RIGHT_HIP]
                right_knee = user.skeleton.joints[nite2.JointType.NITE_JOINT_RIGHT_KNEE]
                right_foot = user.skeleton.joints[nite2.JointType.NITE_JOINT_RIGHT_FOOT]
                left_hip = user.skeleton.joints[nite2.JointType.NITE_JOINT_LEFT_HIP]
                left_knee = user.skeleton.joints[nite2.JointType.NITE_JOINT_LEFT_KNEE]
                left_foot = user.skeleton.joints[nite2.JointType.NITE_JOINT_LEFT_FOOT]
                # 8,9 torso could refer to the spine joint as well, still unsure
                neck = user.skeleton.joints[nite2.JointType.NITE_JOINT_NECK]
                head = user.skeleton.joints[nite2.JointType.NITE_JOINT_HEAD]
                left_shoulder = user.skeleton.joints[nite2.JointType.NITE_JOINT_LEFT_SHOULDER]
                left_elbow = user.skeleton.joints[nite2.JointType.NITE_JOINT_LEFT_ELBOW]
                left_hand = user.skeleton.joints[nite2.JointType.NITE_JOINT_LEFT_HAND]
                right_shoulder = user.skeleton.joints[nite2.JointType.NITE_JOINT_RIGHT_SHOULDER]
                right_elbow = user.skeleton.joints[nite2.JointType.NITE_JOINT_RIGHT_ELBOW]
                right_hand = user.skeleton.joints[nite2.JointType.NITE_JOINT_RIGHT_HAND]

                # confidence = head.positionConfidence
                # print("Head: (x:%dmm, y:%dmm, z:%dmm), confidence: %.2f" % (
                #     head.position.x,
                #     head.position.y,
                #     head.position.z,
                #     confidence))
                confidence = right_hand.positionConfidence
                print("Right hand: (x:%dmm, y:%dmm, z:%dmm), confidence: %.2f" % (
                    right_hand.position.x,
                    right_hand.position.y,
                    right_hand.position.z,
                    confidence))

                time_stamp = timeit.default_timer()
                items.append([torso.position.x,torso.position.y,torso.position.z,
                             right_hip.position.x, right_hip.position.y, right_hip.position.z,
                             right_knee.position.x, right_knee.position.y, right_knee.position.z,
                             right_foot.position.x, right_foot.position.y, right_foot.position.z,
                             left_hip.position.x, left_hip.position.y, left_hip.position.z,
                             left_knee.position.x, left_knee.position.y, left_knee.position.z,
                             left_foot.position.x, left_foot.position.y, left_foot.position.z,
                             neck.position.x, neck.position.y, neck.position.z,
                             head.position.x, head.position.y, head.position.z,
                             left_shoulder.position.x, left_shoulder.position.y, left_shoulder.position.z,
                             left_elbow.position.x, left_elbow.position.y, left_elbow.position.z,
                             left_hand.position.x, left_hand.position.y, left_hand.position.z,
                             right_shoulder.position.x, right_shoulder.position.y, right_shoulder.position.z,
                             right_elbow.position.x, right_elbow.position.y, right_elbow.position.z,
                             right_hand.position.x, right_hand.position.y, right_hand.position.z,
                             time_stamp
                             ])
                np.savetxt('data_kinect.csv',
                           (items),
                           delimiter=',')

    x=4;


nite2.unload()
openni2.unload()