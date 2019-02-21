import os
import time
from pymo.parsers import BVHParser
from pymo.preprocessing import *



class DataLoader:
    def __init__(self, root_path = 'motion_data/'):
        self.path_list = []
        self.total_path_list = []
        self.subject_names = os.listdir(root_path)
        for subject_num in self.subject_names:
            dir_path = os.path.join(root_path, subject_num)
            file_names = os.listdir(dir_path)
            for file_name in file_names:
                self.total_path_list.append(os.path.join(dir_path, file_name))

        self.parser = BVHParser()
        self.mp = MocapParameterizer('position')
        self.label_idx_from = None
        self.label_idx_to = None
        self.joints_to_draw = None
        self.df = None

    def set_bvh_data_path(self, is_train, train_rate = 0.9):
        motion_list= []
        for path in self.toral_path_list :
            is_bvh = path.find('.bvh')
            if is_bvh == -1 :
                continue
            else :
                motion_list.append(path)
        motion_len = len(motion_list)
        train_len = int(motion_len * train_rate)
        if is_train :
            self.path_list = motion_list[:train_len]
        else :
            self.path_list = motion_list[train_len:]

    def merge_velocity_data(self):
        motion_npy_list = []
        input_batch_list = []
        label_batch_list = []
        for path in self.total_path_list:
            is_npy = path.find('.npy')
            if is_npy == -1:
                continue
            else:
                motion_npy_list.append(path)

        total_len = len(motion_npy_list)
        progress = 0

        for path in motion_npy_list:
            start_time = time.time()

            motion = np.load(path)
            pre_motion = np.pad(motion, ((1, 0), (0, 0)), 'constant')
            post_motion = np.pad(motion, ((0, 1), (0, 0)), 'constant')
            velocity_motion = post_motion - pre_motion
            velocity_motion = velocity_motion[1:-1]
            motion_with_velocity = np.concatenate([motion[1:], velocity_motion], axis=1)
            is_input = path.find('input')
            if is_input == -1:
                label_batch_list.append(motion_with_velocity)
            else:
                input_batch_list.append(motion_with_velocity)

            progress = progress + 1
            ptime = time.time() - start_time
            print('[%d/%d] ptime = %.3f' % (progress, total_len, ptime))

        total_input_batch = np.concatenate(input_batch_list, axis=0)
        total_label_batch = np.concatenate(label_batch_list, axis=0)

        return total_input_batch, total_label_batch



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

        total_len = len(self.path_list)
        progress = 0
        for path in self.path_list:
            start_time = time.time()

            parsed_data = self.parser.parse(path)
            positions = self.mp.fit_transform([parsed_data])
            input_batch, label_batch = self.get_pos_batch(positions, down_sample_scale)
            input_batch_list.append(input_batch)
            label_batch_list.append(label_batch)

            progress = progress + 1
            ptime = time.time() - start_time
            print('[%d/%d] ptime = %.3f'%(progress, total_len, ptime))

        total_input_batch = np.concatenate(input_batch_list, axis=0)
        total_label_batch = np.concatenate(label_batch_list, axis=0)

        return total_input_batch, total_label_batch

    def save_total_motion(self, label_idx_from, label_idx_to, down_sample_scale=1):
        self.label_idx_from = label_idx_from
        self.label_idx_to = label_idx_to

        total_len = len(self.path_list)
        progress = 0
        for path in self.path_list:
            start_time = time.time()

            parsed_data = self.parser.parse(path)
            positions = self.mp.fit_transform([parsed_data])
            input_batch, label_batch = self.get_pos_batch(positions, down_sample_scale)
            input_data_path = path[:-4] + '_input.npy'
            label_data_path = path[:-4] + '_label.npy'
            np.save(input_data_path, input_batch)
            np.save(label_data_path, label_batch)

            progress = progress + 1
            ptime = time.time() - start_time
            print('[%d/%d] ptime = %.3f'%(progress, total_len, ptime))