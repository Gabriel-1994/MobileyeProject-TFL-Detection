import numpy as np
import pickle
import matplotlib.pyplot as plt
import MeasureDistance

def visualize(prev_container, curr_container, focal, pp,prev_frame_id,curr_frame_id,axs):
    norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = MeasureDistance.prepare_3D_data(prev_container, curr_container, focal, pp)
    norm_rot_pts = MeasureDistance.rotate(norm_prev_pts, R)
    rot_pts = MeasureDistance.unnormalize(norm_rot_pts, focal, pp)
    foe = np.squeeze(MeasureDistance.unnormalize(np.array([norm_foe]), focal, pp))
    axs[2].imshow(curr_container.img)
    axs[2].set_ylabel('distance')
    curr_p = curr_container.traffic_light
    axs[2].plot(curr_p[:, 0], curr_p[:, 1], 'b+')
    for i in range(len(curr_p)):
        if curr_container.valid[i]:
            axs[2].text(curr_p[i, 0], curr_p[i, 1],
                          r'{0:.1f}'.format(curr_container.traffic_lights_3d_location[i, 2]), color='r')
    axs[2].plot(foe[0], foe[1], 'r+')
    axs[2].plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')

class CurrContainer(object):
    def __init__(self, img_path):
        self.img = plt.imread(img_path)
        self.path = img_path
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []

class PrevContainer(object):
    def __init__(self, img_path):
        self.img = plt.imread(img_path)
        self.path = img_path
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []

def read_data_and_run(prev_frame,curr_frame,axs,pkl_path):
    # read data and run
    prev_frame_path = prev_frame.path.split("\\")[-1]
    prev_frame_id = prev_frame_path.split("_")[-2]
    prev_frame_id = (prev_frame_id[4:])
    curr_frame_path = curr_frame.path.split("\\")[-1]
    curr_frame_id = curr_frame_path.split("_")[-2]
    curr_frame_id = (curr_frame_id[4:])
    # Read the pkl path
    with open(pkl_path, 'rb') as pklfile:
        data = pickle.load(pklfile, encoding='latin1')
    focal = data['flx']
    pp = data['principle_point']
    EM = np.eye(4)
    for i in range( int(curr_frame_id), int(prev_frame_id)):
        EM = np.dot(data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
    curr_frame.EM = EM
    curr_frame = MeasureDistance.calc_TFL_dist(prev_frame, curr_frame, focal, pp)
    visualize(prev_frame, curr_frame, focal, pp,prev_frame_id,curr_frame_id,axs)
