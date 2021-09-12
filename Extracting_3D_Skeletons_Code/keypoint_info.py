# My findings: indexes 0 to 16 of cls_keyps variable(found in "keypoints" file)
# Note the only relevant set of keypoints are the one's with a score in
# the clc_boxes variable with more than 0.9 in the score entry (index 4)
# ( as per the documentation, vis.py, line 222 to line 246) It appears
# Like the score and logit entries of he key points (entry 3 and 2 respectively) are not
# used so don't bother with them, even though it would have a similar effects if
# you used the mean score of all body parts and threshold to some value (like 0.02),
# Because the scores tend to be higher for body parts than for objects that don't
# have body parts

'nose', # 0
'left_eye',
'right_eye',
'left_ear',
'right_ear',
'left_shoulder',
'right_shoulder',
'left_elbow',
'right_elbow',
'left_wrist',
'right_wrist',
'left_hip',
'right_hip',
'left_knee',
'right_knee',
'left_ankle',
'right_ankle' # 16