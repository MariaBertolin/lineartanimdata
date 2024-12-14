import numpy as np


masks_points = []

#positive points
mask0_pos_x=900
mask0_pos_y=900

mask0_pos3_x=550 #apple
mask0_pos3_y=700

mask0_pos5_x=355 #apple
mask0_pos5_y=245

masks_points.append([np.array([[mask0_pos_x, mask0_pos_y],[mask0_pos5_x, mask0_pos5_y],[mask0_pos3_x, mask0_pos3_y]], dtype=np.float32), np.array([1,1,1], np.int32)])

