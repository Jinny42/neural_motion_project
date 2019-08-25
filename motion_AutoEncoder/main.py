import MotionAE
import os
import shutil
import data_processing
import numpy as np
import skeleton_plot

def model_choose(model_kind) :
    batch_size = 512
    learning_rate = 0.0002
    momentum = 0.9
    weight_decay = 0.0001

    MoAE = MotionAE.MotionAE()
    MoAE.train(10000, model_kind=model_kind, input_size=batch_size, lr=learning_rate, wd=weight_decay, momentum=momentum,
              done_epoch=0, gpu_num=2, split_num=1)

    # MoAE.train(100, model_kind=model_kind, input_size=batch_size, lr=learning_rate, wd=weight_decay, momentum=momentum,
    #           done_epoch=0, sampling_step=sampling_step)
    #
    # MoAE.train(100, model_kind=model_kind, input_size=batch_size, lr=learning_rate / 10, wd=weight_decay, momentum=momentum,
    #           done_epoch=100, sampling_step=sampling_step)
    #
    # MoAE.train(100, model_kind=model_kind, input_size=batch_size, lr=learning_rate / 100, wd=weight_decay, momentum=momentum,
    #           done_epoch=200, sampling_step=sampling_step)

def inference(model_kind, motion_name) :
    batch_size = 512
    learning_rate = 0.0002
    momentum = 0.9
    weight_decay = 0.00001

    os.makedirs('preprocessed_data/clips/for_inference/'+motion_name)
    Maker = data_processing.ClipMaker(target_path='preprocessed_data/clips/for_inference/' + motion_name)
    Maker.set_manual_path(['preprocessed_data/ordered_position\\' + motion_name + '.npy'])
    Maker.make_rel_pos_clips(downsample_rate=1)

    MoAE = MotionAE.MotionAE()
    MoAE.inference(model_kind=model_kind, input_size=batch_size, lr=learning_rate, wd=weight_decay, momentum=momentum, input_path='preprocessed_data/clips/for_inference/' + motion_name)

    pos_data = np.load('preprocessed_data/ordered_position\\' + motion_name +'.npy')  # [Frame, Joint, Channel] : [Frame, 11, 3]
    len_pos = len(pos_data)
    clip_data = np.load(
        'preprocessed_data/clips/for_inference/' + motion_name + '\\output.npy')  # [Frame - (self.unit_frame -1), Joint, Channel, Reference Joint] : [Frame - 63, 2, 3, 3]

    new_pos = Maker.clip2pos_rel(pos_data, clip_data[:len_pos - (Maker.unit_frame - 1)])

    np.save('preprocessed_data/clips/for_inference/' + motion_name + '\\position.npy', new_pos)

    joint_order = ('lhand', 'LeftShoulder', 'LeftToe', 'LeftKnee',
                   'Hips', 'Chest2', 'Head',
                   'RightKnee', 'RightToe', 'RightShoulder', 'rhand',
                   'inf_LeftToe', 'inf_LeftKnee',
                   'inf_RightKnee', 'inf_RightToe')

    data = np.load('preprocessed_data/clips/for_inference/' + motion_name + '/position.npy')
    plotter = skeleton_plot.Skeleton_inference_plot(data, joint_order)
    plotter.show()

def inference_raw(model_kind, motion_name) :
    batch_size = 512
    learning_rate = 0.0002
    momentum = 0.9
    weight_decay = 0.00001

    isdir = os.path.isdir('preprocessed_data/clips/for_inference/'+motion_name)
    if isdir :
        shutil.rmtree('preprocessed_data/clips/for_inference/'+motion_name)
        os.makedirs('preprocessed_data/clips/for_inference/' + motion_name)
    else :
        os.makedirs('preprocessed_data/clips/for_inference/'+motion_name)
    Maker = data_processing.ClipMaker(target_path='preprocessed_data/clips/for_inference/' + motion_name)
    Maker.set_manual_path(['preprocessed_data/ordered_position\\' + motion_name + '.npy'])
    Maker.make_raw_pos_clips(downsample_rate=1)
    #
    MoAE = MotionAE.MotionAE()
    MoAE.inference2(model_kind=model_kind, input_size=batch_size, lr=learning_rate, wd=weight_decay, momentum=momentum, input_path='preprocessed_data/clips/for_inference/' + motion_name)

    pos_data = np.load('preprocessed_data/ordered_position\\' + motion_name +'.npy')  # [Frame, Joint, Channel] : [Frame, 11, 3]
    len_pos = len(pos_data)
    clip_data = np.load(
        'preprocessed_data/clips/for_inference/' + motion_name + '\\output.npy')  # [Frame - (self.unit_frame -1), Joint, Channel, Reference Joint] : [Frame - 63, 2, 3, 3]

    new_pos = Maker.clip2pos(pos_data, clip_data[:len_pos - (Maker.unit_frame - 1)])

    # np.save(motion_name + '_position.npy', new_pos)

    np.save('preprocessed_data/clips/for_inference/' + motion_name + '\\position.npy', new_pos)

    joint_order = ('lhand', 'LeftShoulder', 'LeftToe', 'LeftKnee',
                   'Hips', 'Chest2', 'Head',
                   'RightKnee', 'RightToe', 'RightShoulder', 'rhand',
                   'inf_LeftToe', 'inf_LeftKnee',
                   'inf_RightKnee', 'inf_RightToe')

    data = np.load('preprocessed_data/clips/for_inference/' + motion_name + '/position.npy')
    plotter = skeleton_plot.Skeleton_inference_plot(data, joint_order)
    # plotter.show(motion_name + '.gif')
    plotter.show()


# model_choose(2)
data_processing.rotation_tranform('90_02', np.pi*7/5)
inference_raw(2, '90_02_rotated')

