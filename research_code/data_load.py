import numpy as np
import os
from pymo.parsers import BVHParser
from pymo.preprocessing import *


class DataLoader:

    def __init__(self):
        self.path_list = None
        self.parser = BVHParser()
        self.mp = MocapParameterizer('position')
        self.label_idx_from = None
        self.label_idx_to = None
        self.joints_to_draw = None
        self.df = None

    def set_subject_path(self, subject_list):
        self.path_list = []
        for subject_num in subject_list:
            dir_path = 'motion_data/' + str(subject_num) + '/'
            file_names = os.listdir(dir_path)
            for file_name in file_names :
                self.path_list.append(os.path.join(dir_path, file_name))

    def get_pos_list(self, frame):
        pos_list = []

        for joint in self.joints_to_draw:
            parent_x = self.df['%s_Xposition' % joint][frame]
            parent_y = self.df['%s_Yposition' % joint][frame]
            parent_z = self.df['%s_Zposition' % joint][frame]

            pos_list.append(parent_x)
            pos_list.append(parent_y)
            pos_list.append(parent_z)

            # children_to_draw = [c for c in mocap_track.skeleton[joint]['children'] if c in joints_to_draw]
            # for c in children_to_draw:
            #     child_x = df['%s_Xposition' % c][frame]
            #     child_y = df['%s_Yposition' % c][frame]
            #     child_z = df['%s_Zposition' % c][frame]
            #
            #     pos_list.append(child_x)
            #     pos_list.append(child_y)
            #     pos_list.append(child_z)

        return pos_list

    def get_pos_batch(self, pos_data, down_sample_scale):
        batch_size = pos_data[0].values.shape[0] // down_sample_scale
        label_len = self.label_idx_to - self.label_idx_from

        input_batch = np.empty((batch_size, 114-label_len), dtype=np.float32)
        label_batch = np.empty((batch_size, label_len), dtype=np.float32)

        self.joints_to_draw = pos_data[0].skeleton.keys()
        self.df = pos_data[0].values

        for i in range(batch_size):
            pos_list = self.get_pos_list(frame=i * down_sample_scale)

            label_batch[i] = np.asarray(pos_list[self.label_idx_from:self.label_idx_to])
            input_batch[i] = np.asarray(pos_list[:self.label_idx_from] + pos_list[self.label_idx_to:])

        return input_batch, label_batch

    def get_single_motion(self, motion_num, label_idx_from, label_idx_to, down_sample_scale=1):
        idx = motion_num
        self.label_idx_from = label_idx_from
        self.label_idx_to = label_idx_to

        path = self.path_list[idx]
        parsed_data = self.parser.parse(path)
        positions = self.mp.fit_transform([parsed_data])
        input_batch, label_batch = self.get_pos_batch(positions, down_sample_scale)
        return input_batch, label_batch

    def get_multi_motion(self, motion_list, label_idx_from, label_idx_to, down_sample_scale=1):
        self.label_idx_from = label_idx_from
        self.label_idx_to = label_idx_to

        input_batch_list = []
        label_batch_list = []

        for idx in motion_list :
            path = self.path_list[idx]
            parsed_data = self.parser.parse(path)
            positions = self.mp.fit_transform([parsed_data])
            input_batch, label_batch = self.get_pos_batch(positions, down_sample_scale)
            input_batch_list.append(input_batch)
            label_batch_list.append(label_batch)

        multi_input_batch = np.concatenate(input_batch_list, axis=0)
        multi_label_batch = np.concatenate(label_batch_list, axis=0)

        return multi_input_batch, multi_label_batch

    def get_total_motion(self, label_idx_from, label_idx_to, down_sample_scale=1):
        self.label_idx_from = label_idx_from
        self.label_idx_to = label_idx_to

        input_batch_list = []
        label_batch_list = []

        for path in self.path_list:
            parsed_data = self.parser.parse(path)
            positions = self.mp.fit_transform([parsed_data])
            input_batch, label_batch = self.get_pos_batch(positions, down_sample_scale)
            input_batch_list.append(input_batch)
            label_batch_list.append(label_batch)

        total_input_batch = np.concatenate(input_batch_list, axis=0)
        total_label_batch = np.concatenate(label_batch_list, axis=0)

        return total_input_batch, total_label_batch

