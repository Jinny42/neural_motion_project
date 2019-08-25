import os
import time
from pymo.parsers import BVHParser
from pymo.preprocessing import *
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, root_path='motion_data/', target_path='preprocessed_data/raw_position'):
        self.total_path_list = []
        self.subject_names = os.listdir(root_path)
        for subject_num in self.subject_names:
            dir_path = os.path.join(root_path, subject_num)
            file_names = os.listdir(dir_path)
            for file_name in file_names:
                self.total_path_list.append(os.path.join(dir_path, file_name))
        self.target_path = target_path

        self.parser = BVHParser()
        self.mp = MocapParameterizer('position')
        self.joints_to_draw = ('Hips', 'RightKnee', 'RightToe', 'Chest2', 'Head', 'LeftShoulder', 'lhand', 'RightShoulder', 'rhand', 'LeftKnee', 'LeftToe')
        self.df = None
        self.min_frame = 100

    def get_pos_of_a_frame(self, frame):
        pos_batch = np.empty((11, 3), dtype=np.float32) ##[joint_num, channel_num(xyz)] = [11, 3]

        for idx, joint in enumerate(self.joints_to_draw):
            parent_x = self.df['%s_Xposition' % joint][frame]
            parent_y = self.df['%s_Yposition' % joint][frame]
            parent_z = self.df['%s_Zposition' % joint][frame]

            pos_batch[idx, 0] = parent_x
            pos_batch[idx, 1] = parent_y
            pos_batch[idx, 2] = parent_z

            # children_to_draw = [c for c in mocap_track.skeleton[joint]['children'] if c in joints_to_draw]
            # for c in children_to_draw:
            #     child_x = df['%s_Xposition' % c][frame]
            #     child_y = df['%s_Yposition' % c][frame]
            #     child_z = df['%s_Zposition' % c][frame]
            #
            #     pos_list.append(child_x)
            #     pos_list.append(child_y)
            #     pos_list.append(child_z)

        return pos_batch

    def get_pos_batch(self, pos_data, down_sample_scale):
        batch_len = pos_data[0].values.shape[0] // down_sample_scale

        total_batch = np.empty((batch_len, 11, 3), dtype=np.float32) ##[frame_num, joint_num, channel_num(xyz)] = [f, 11, 3]

        # self.joints_to_draw = pos_data[0].skeleton.keys()
        self.df = pos_data[0].values

        for i in range(batch_len):
            frame_batch = self.get_pos_of_a_frame(frame=i * down_sample_scale)

            total_batch[i] = frame_batch

        return total_batch

    def save_total_motion(self, down_sample_scale=1):
        total_len = len(self.total_path_list)
        progress = 0
        processed_motion_num = 0
        filtered_motion_num = 0

        for path in self.total_path_list:
            start_time = time.time()

            start_idx = path.find('\\') + 1
            end_idx = path.find('.bvh')
            output_file_name = path[start_idx:end_idx] + '.npy'

            output_path = os.path.join(self.target_path, output_file_name)

            parsed_data = self.parser.parse(path)
            positions = self.mp.fit_transform([parsed_data])

            batch_size = positions[0].values.shape[0] // down_sample_scale
            if self.min_frame > batch_size :
                print(output_file_name + 'is too short : ', batch_size, ' frames')
                filtered_motion_num = filtered_motion_num + 1
                continue
            else :
                processed_motion_num = processed_motion_num + 1

            single_motion_batch = self.get_pos_batch(positions, down_sample_scale)

            np.save(output_path, single_motion_batch)

            progress = progress + 1
            ptime = time.time() - start_time
            print('[%d/%d] ptime = %.3f' % (progress, total_len, ptime))

        print('processed_motion_num = ' , processed_motion_num, '/ filtered_motion_num = ', filtered_motion_num)


class ClipMaker:
    def __init__(self, root_path='preprocessed_data/ordered_position',
                 target_path='preprocessed_data/clips/ordered_clips',
                 joint_order= ('lhand', 'LeftShoulder', 'LeftToe', 'LeftKnee',
                        'Hips', 'Chest2', 'Head',
                        'RightKnee', 'RightToe', 'RightShoulder','rhand')):
        self.total_path_list = []
        file_names = os.listdir(root_path)
        for file_name in file_names:
            self.total_path_list.append(os.path.join(root_path, file_name))
        self.total_len = len(self.total_path_list)
        self.target_path = target_path

        self.unit_frame = 64

        # indexes of joints
        self.hips_idx = joint_order.index('Hips')
        self.rknee_idx = joint_order.index('RightKnee')
        self.rtoe_idx = joint_order.index('RightToe')
        self.chest_idx = joint_order.index('Chest2')
        self.head_idx = joint_order.index('Head')
        self.lshoulder_idx = joint_order.index('LeftShoulder')
        self.lhand_idx = joint_order.index('lhand')
        self.rshoulder_idx = joint_order.index('RightShoulder')
        self.rhand_idx = joint_order.index('rhand')
        self.lknee_idx = joint_order.index('LeftKnee')
        self.ltoe_idx = joint_order.index('LeftToe')

    def make_raw_pos_clips(self, downsample_rate= 15):
        min_value_clip = np.inf
        max_value_clip = -np.inf

        for path in self.total_path_list:
            motion = np.load(path)
            motion_len = len(motion)

            total_submotion_num = (motion_len - self.unit_frame) // downsample_rate + 1

            for i in range(total_submotion_num) :
                unit_motion = motion[0 + i * downsample_rate:self.unit_frame + i * downsample_rate] #[64, 11, 3]

                start_idx = path.find('\\') + 1
                end_idx = path.find('.npy')
                output_file_name = path[start_idx:end_idx] + '_' + '%4d'%(i+1) +'.npy'
                output_path = os.path.join(self.target_path, output_file_name)

                np.save(output_path, unit_motion)

                min_value_clip = min(min_value_clip, np.amin(unit_motion))
                max_value_clip = max(max_value_clip, np.amax(unit_motion))

        print('min : ', min_value_clip, ' max : ', max_value_clip)

    def make_rel_pos_clips(self, downsample_rate= 15):
        min_value_clip = np.inf
        max_value_clip = -np.inf

        for path in self.total_path_list:
            motion = np.load(path)
            motion_len = len(motion)

            total_submotion_num = (motion_len - self.unit_frame) // downsample_rate + 1

            for i in range(total_submotion_num) :
                unit_motion = motion[0 + i * downsample_rate:self.unit_frame + i * downsample_rate] #[64, 11, 3]
                output_batch = np.empty((64, 11, 3),
                                        dtype=np.float32)  # [height(frame), width(joint), channel(axis), reference joint]

                ## get projection of hip to base
                #### get hip pos
                reference_col = unit_motion[:, self.hips_idx, :] #[64, 3]
                #### set y value of hip to 0
                reference_col[:, 1] = 0
                reference_col = np.expand_dims(reference_col, axis = 1) #[64, 1, 3]
                reference_tile = np.tile(reference_col, (1, 11, 1)) #[64, 11, 3]
                output_batch = unit_motion - reference_tile # [64, 11, 3]  :  [height(frame), width(joint), channel(axis), reference joint]

                start_idx = path.find('\\') + 1
                end_idx = path.find('.npy')
                output_file_name = path[start_idx:end_idx] + '_' + '%4d'%(i+1) +'.npy'
                output_path = os.path.join(self.target_path, output_file_name)

                np.save(output_path, output_batch)

                min_value_clip = min(min_value_clip, np.amin(output_batch))
                max_value_clip = max(max_value_clip, np.amax(output_batch))

        print('min : ', min_value_clip, ' max : ', max_value_clip)

    def normalize_clips(self, min, max):
        total_clip_path_list = []
        file_names = os.listdir(self.target_path)
        for file_name in file_names:
            total_clip_path_list.append(os.path.join(self.target_path, file_name))
        total_len = len(total_clip_path_list)

        for path in total_clip_path_list :
            clip = np.load(path)
            normalized_clip = self.normalize(clip)
            # plt.imshow(normalized_clip[:, :, :, 1].astype(np.uint))
            # plt.show()
            np.save(path, normalized_clip)


    # # for comparison in exp2 (image's center inpainting, image's edge inpainting)
    # def idx_swap(self, motion_mat, joint_idx1, joint_idx2):
    #     motion_mat[:, joint_idx1,:], motion_mat[:, joint_idx2,:] = motion_mat[:, joint_idx2,:], motion_mat[:, joint_idx1,:]
    #
    #     return motion_mat

    def clip2pos(self, pos_data, clip_data):
        # pos_data : [Frame, 11, 3]
        # clip_data : [Frame - 63, 4, 3]

        # declare output array
        frame_len = len(pos_data)
        output_data = np.zeros([frame_len, 11 + 4, 3],
                               dtype=np.float32)

        # copy origin area [Frame, 11, 3]
        output_data[:, :11, :] = pos_data

        output_data[(self.unit_frame-1):, 11:, :] = clip_data

        return output_data

    def clip2pos_rel(self, pos_data, clip_data):
        # pos_data : [Frame, 11, 3]
        # clip_data : [Frame - 63, 4, 3]

        # declare output array
        frame_len = len(pos_data)
        output_data = np.zeros([frame_len, 11 + 4, 3],
                               dtype=np.float32)

        # copy origin area [Frame, 11, 3]
        output_data[:, :11, :] = pos_data

        # clip value is 'displacement'. for converting it to position, add it to reference's position.
        ref_pos = np.expand_dims(pos_data[(self.unit_frame - 1):, self.hips_idx, :], axis=1)  # [Frame - 63, 1, 3]
        ref_pos[:,:,1] = 0
        ref_pos = np.tile(ref_pos, (1, 4, 1))  # [Frame - 63 , 4, 3]

        pos_converted = clip_data + ref_pos

        output_data[(self.unit_frame-1):, 11:, :] = pos_converted

        return output_data

    def set_manual_path(self, path_list):
        self.total_path_list = path_list

    def normalize(self, data):
        return (data +30) / 60 * 255

    def unnormalize(self, data):
        return (data / 255 * 60) - 30




#huristically re-order joint's index
def exp1_reorder_idx(root_path='preprocessed_data/raw_position', target_path='preprocessed_data/ordered_position'):
    total_path_list = []
    file_names = os.listdir(root_path)
    for file_name in file_names:
        total_path_list.append(os.path.join(root_path, file_name))

    input_joint_idx = ('Hips', 'RightKnee', 'RightToe', 'Chest2', 'Head', 'LeftShoulder', 'lhand', 'RightShoulder',
                      'rhand', 'LeftKnee', 'LeftToe')

    output_joint_idx = ('lhand', 'LeftShoulder', 'LeftToe', 'LeftKnee',
                        'Hips', 'Chest2', 'Head',
                        'RightKnee', 'RightToe', 'RightShoulder','rhand')

    for path in total_path_list:
        input_batch = np.load(path)
        batch_len = len(input_batch)
        output_batch = np.empty((batch_len, 11, 3), dtype=np.float32)

        start_idx = path.find('\\') + 1
        output_file_name = path[start_idx:]
        output_path = os.path.join(target_path, output_file_name)

        for idx, joint in enumerate(input_joint_idx):
            new_idx = output_joint_idx.index(joint)
            output_batch[:, new_idx, :] = input_batch[:, idx, :]

        np.save(output_path, output_batch)

def rotation_tranform(motion_name, radians_):
    path = 'preprocessed_data/ordered_position\\' + motion_name + '.npy'
    position = np.load(path)
    transform_mat = np.array([
        [np.cos(radians_), 0, np.sin(radians_)],
        [0, 1, 0],
        [-np.sin(radians_), 0, np.cos(radians_)]
    ])

    pos_transformed = np.matmul(position, np.transpose(transform_mat))

    np.save('preprocessed_data/ordered_position\\' + motion_name + '_rotated.npy', pos_transformed)





# Maker = ClipMaker(target_path='preprocessed_data/clips/rel_pos_clips')
# Maker.make_rel_pos_clips(downsample_rate=15)
