from pymo.parsers import BVHParser
from pymo.preprocessing import *
from plot_functions import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_pos_list(mocap_track, frame, data=None, joints=None):
    if joints is None:
        joints_to_draw = mocap_track.skeleton.keys()
    else:
        joints_to_draw = joints

    if data is None:
        df = mocap_track.values
    else:
        df = data

    pos_list = []

    # num = 0

    for joint in joints_to_draw:
        parent_x = df['%s_Xposition' % joint][frame]
        parent_y = df['%s_Yposition' % joint][frame]
        parent_z = df['%s_Zposition' % joint][frame]

        pos_list.append(parent_x)
        pos_list.append(parent_y)
        pos_list.append(parent_z)

        # print('joint : ', joint, ' ', num)
        #
        # num = num +1

        children_to_draw = [c for c in mocap_track.skeleton[joint]['children'] if c in joints_to_draw]

        for c in children_to_draw:
            child_x = df['%s_Xposition' % c][frame]
            child_y = df['%s_Yposition' % c][frame]
            child_z = df['%s_Zposition' % c][frame]

            pos_list.append(child_x)
            pos_list.append(child_y)
            pos_list.append(child_z)

            # print('child : ', c, ' ', num)
            #
            # num = num + 1

    return pos_list

def get_pos_batch(pos_data):
    batch_size = positions[0].values.shape[0]
    pos_batch = np.empty((batch_size, 222), dtype=np.float32)
    hip_label = np.empty((batch_size, 3), dtype=np.float32)
    for i in range(batch_size):
        pos_batch[i] = np.asarray(get_pos_list(pos_data[0], frame=i)[3:])
        hip_label[i] = np.asarray(get_pos_list(pos_data[0], frame=i)[:3])


    return pos_batch, hip_label


def make_model(X, isTrain):
    W1 = tf.get_variable("W1", shape=[222, 3], dtype=np.float32,
                                  initializer=tf.random_normal_initializer(0, tf.sqrt(2/222)))# He initialization
    L1 = tf.matmul(X, W1)
    # L1 = tf.layers.batch_normalization(L1, training=isTrain)
    # L1 = tf.nn.relu(L1)

    # W2 = tf.get_variable("W2", shape=[512, 512], dtype=np.float32,
    #                      initializer=tf.random_normal_initializer(0, tf.sqrt(2/512)))
    # L2 = tf.matmul(L1, W2)
    # L2 = tf.layers.batch_normalization(L2, training=isTrain)
    # L2 = tf.nn.relu(L2)
    #
    # W3 = tf.get_variable("W3", shape=[512, 1024], dtype=np.float32,
    #                      initializer=tf.random_normal_initializer(0, tf.sqrt(2/1024)))
    # L3 = tf.matmul(L2, W3)
    # L3 = tf.layers.batch_normalization(L3, training=isTrain)
    # L3 = tf.nn.relu(L3)
    #
    # W4 = tf.get_variable("W4", shape=[1024, 1024], dtype=np.float32,
    #                      initializer=tf.random_normal_initializer(0, tf.sqrt(2 / 1024)))
    # L4 = tf.matmul(L3, W4)
    # L4 = tf.layers.batch_normalization(L4, training=isTrain)
    # L4 = tf.nn.relu(L4)
    #
    # W5 = tf.get_variable("W5", shape=[1024, 2048], dtype=np.float32,
    #                      initializer=tf.random_normal_initializer(0, tf.sqrt(2 / 1024)))
    # L5 = tf.matmul(L4, W5)
    # L5 = tf.layers.batch_normalization(L5, training=isTrain)
    # L5 = tf.nn.relu(L5)
    #
    # W6 = tf.get_variable("W6", shape=[2048, 2048], dtype=np.float32,
    #                      initializer=tf.random_normal_initializer(0, tf.sqrt(2 / 1024)))
    # L6 = tf.matmul(L5, W6)
    # L6 = tf.layers.batch_normalization(L6, training=isTrain)
    # L6 = tf.nn.relu(L6)
    #
    # W7 = tf.get_variable("W7", shape=[2048, 3], dtype=np.float32,
    #                      initializer=tf.random_normal_initializer(0, tf.sqrt(2/1024)))
    # L7 = tf.matmul(L6, W7)

    return L1

def make_train_graph(input, label, is_training):
    logit = make_model(input, is_training)
    loss_L2 = tf.reduce_sum(tf.nn.l2_loss(logit - label))
    # reduce_mean 대신 reduce_sum을 사용했을때 더 잘 수렴했음
    # batch size와 output node 수에 비례해 scale을 해줬다고 볼 수 있는데
    # facebook에서 발표한 논문에서도 이게 꽤 괜찮은 테크닉이라고 소개하고 있음.
    # 논문링크 : https://research.fb.com/publications/accurate-large-minibatch-sgd-training-imagenet-in-1-hour/
    # 관련링크 : https://stats.stackexchange.com/questions/201452/is-it-common-practice-to-minimize-the-mean-loss-over-the-batches-instead-of-the/201540#201540

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = tf.train.AdamOptimizer(0.0002).minimize(loss_L2)

    return train_op, loss_L2, logit

def train(input, label, max_epoch) :
    #####Make placeholder
    ### Input
    X = tf.placeholder(tf.float32, [None, 222])
    ### Label
    Y = tf.placeholder(tf.float32, [None, 3])
    ### Is training
    is_training = tf.placeholder(tf.bool)

    ##### Make Graph
    train_op, loss_L2, logit_ = make_train_graph(X, Y, is_training)

    ##### Run Session
    sess = tf.Session()

    ##### Check the Checkpoint
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    for epoch in range(max_epoch):
        ###########################################################################################################

        ########################################################################################################
        ##Update
        _, loss, logit = sess.run([train_op, loss_L2, logit_],
                             feed_dict={X: input, Y: label, is_training: True})

        if epoch % 100 == 0 :
            saver.save(sess, './model/model.ckpt')
            print('saved')

        print('[%d/%d] -  loss: %.3f'
              % ((epoch + 1), max_epoch, loss))

def inference(input, label) :
    ##### Loss Log txt file
    log_txt = open('inference_loss.txt', 'a')

    #####Make placeholder
    ### Input
    X = tf.placeholder(tf.float32, [None, 222])
    ### Label
    Y = tf.placeholder(tf.float32, [None, 3])
    ### Is training
    is_training = tf.placeholder(tf.bool)

    ##### Make Graph
    train_op, loss_L2, logit_ = make_train_graph(X, Y, is_training)

    ##### Run Session
    sess = tf.Session()

    ##### Check the Checkpoint
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    for i in range(400+len(input[400:])):
        loss, logit = sess.run([loss_L2, logit_],
                             feed_dict={X: np.expand_dims(input[i], axis=0),
                                        Y: np.expand_dims(label[i], axis=0),
                                        is_training: False})
        log_txt.write('\n[%d] - %.3f'%(i, np.sqrt(loss*3)))

    log_txt.close()
    print('loss: %.3f'%(loss))

def cross_val(input, label, max_epoch) :
    ##### Histogram for Visualizing
    log_ED = {}
    log_ED['train_ED'] = []
    log_ED['val_ED'] = []

    ##### Loss Log txt file
    log_txt = open('loss.txt', 'a')

    #####Make placeholder
    ### Input
    X = tf.placeholder(tf.float32, [None, 222])
    ### Label
    Y = tf.placeholder(tf.float32, [None, 3])
    ### Is training
    is_training = tf.placeholder(tf.bool)

    ##### Make Graph
    train_op, loss_L2, logit_ = make_train_graph(X, Y, is_training)

    ##### Run Session
    sess = tf.Session()

    ##### Check the Checkpoint
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    for epoch in range(max_epoch):
        ###########################################################################################################

        ########################################################################################################
        ##Update
        _, loss, logit = sess.run([train_op, loss_L2, logit_],
                             feed_dict={X: input[:400], Y: label[:400], is_training: True})

        mean_of_ED = np.sqrt(3 * loss / input[:400].shape[0])
        log_txt.write('\n\n[%d/%d] -  loss: %.3f, mean of ED: %.3f'
               % ((epoch + 1), max_epoch, loss, mean_of_ED))
        # print('[%d/%d] -  loss: %.3f, mean of ED: %.3f'
        #       % ((epoch + 1), max_epoch, loss, mean_of_ED))
        log_ED['train_ED'].append(mean_of_ED)

        if epoch % 1 == 0 :
            loss, logit = sess.run([loss_L2, logit_],
                                   feed_dict={X: input[400:], Y: label[400:], is_training: False})

            mean_of_ED = np.sqrt(3 * loss / input[400:].shape[0])
            log_txt.write('\nloss: %.3f, mean of ED: %.3f'
                   % (loss, mean_of_ED))
            # print('loss: %.3f, mean of ED: %.3f'
            #       % (loss, mean_of_ED))
            log_ED['val_ED'].append(mean_of_ED)

            epoch_ED_plot(log_ED, save=True, path='train_hist.png')

    log_txt.close()
    saver.save(sess, './model/model.ckpt')



parser = BVHParser()

parsed_data = parser.parse('motion_data/69/69_01.bvh')

mp = MocapParameterizer('position')

positions = mp.fit_transform([parsed_data])

# draw_stickfigure3d(positions[0], 1)
# plt.show()

position_batch, hip_label = get_pos_batch(positions)

# cross_val(position_batch, hip_label, 10000)
# train(position_batch, hip_label, 10000)
inference(position_batch, hip_label)