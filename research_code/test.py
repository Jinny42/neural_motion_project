from plot_functions import *
import data_load
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def make_model1(X, isTrain):
    W1 = tf.get_variable("W1", shape=[111, 3], dtype=np.float32,
                                  initializer=tf.random_normal_initializer(0, tf.sqrt(2/111)))# He initialization
    L1 = tf.matmul(X, W1)

    return L1

def make_model2(X, isTrain):
    W1 = tf.get_variable("W1", shape=[111, 256], dtype=np.float32,
                                  initializer=tf.random_normal_initializer(0, tf.sqrt(2/111)))# He initialization
    L1 = tf.matmul(X, W1)
    L1 = tf.layers.batch_normalization(L1, training=isTrain)
    L1 = tf.nn.relu(L1)

    W2 = tf.get_variable("W2", shape=[256, 512], dtype=np.float32,
                         initializer=tf.random_normal_initializer(0, tf.sqrt(2/256)))
    L2 = tf.matmul(L1, W2)
    L2 = tf.layers.batch_normalization(L2, training=isTrain)
    L2 = tf.nn.relu(L2)


    W3 = tf.get_variable("W3", shape=[512, 1024], dtype=np.float32,
                         initializer=tf.random_normal_initializer(0, tf.sqrt(2/512)))
    L3 = tf.matmul(L2, W3)
    L3 = tf.layers.batch_normalization(L3, training=isTrain)
    L3 = tf.nn.relu(L3)

    W4 = tf.get_variable("W4", shape=[1024, 3], dtype=np.float32,
                         initializer=tf.random_normal_initializer(0, tf.sqrt(2 / 1024)))
    L4 = tf.matmul(L3, W4)

    return L4

def make_train_graph(input, label, is_training):
    logit = make_model2(input, is_training)

    loss_L2 = tf.pow(logit - label, 2)
    sum_L2 = tf.reduce_sum(loss_L2)
    loss_ED = tf.sqrt(tf.reduce_sum(loss_L2, axis=1))

    # reduce_mean 대신 reduce_sum을 사용했을때 더 잘 수렴했음
    # batch size와 output node 수에 비례해 scale을 해줬다고 볼 수 있는데
    # facebook에서 발표한 논문에서도 이게 꽤 괜찮은 테크닉이라고 소개하고 있음.
    # 논문링크 : https://research.fb.com/publications/accurate-large-minibatch-sgd-training-imagenet-in-1-hour/
    # 관련링크 : https://stats.stackexchange.com/questions/201452/is-it-common-practice-to-minimize-the-mean-loss-over-the-batches-instead-of-the/201540#201540

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = tf.train.AdamOptimizer(0.0002).minimize(sum_L2)

    return train_op, sum_L2, logit, loss_ED

def train(input, label, max_epoch) :
    #####Make placeholder
    ### Input
    X = tf.placeholder(tf.float32, [None, 111])
    ### Label
    Y = tf.placeholder(tf.float32, [None, 3])
    ### Is training
    is_training = tf.placeholder(tf.bool)

    ##### Make Graph
    train_op, sum_L2, logit_, loss_ED = make_train_graph(X, Y, is_training)

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
        _, ED = sess.run([train_op, tf.reduce_mean(loss_ED)],
                             feed_dict={X: input, Y: label, is_training: True})

        if epoch % 100 == 0 :
            saver.save(sess, './model/model.ckpt')
            print('saved')

        print('[%d/%d] - mean ED: %.3f'
              % ((epoch + 1), max_epoch, ED))

def inference() :
    ##### Data Loader
    DataLoader = data_load.DataLoader('motion_data/69/')
    input_batch, label_batch = DataLoader.get_single_motion(1, 0, 3)

    ##### Loss Log txt file
    log_txt = open('inference_loss.txt', 'a')

    #####Make placeholder
    ### Input
    X = tf.placeholder(tf.float32, [None, 111])
    ### Label
    Y = tf.placeholder(tf.float32, [None, 3])
    ### Is training
    is_training = tf.placeholder(tf.bool)

    ##### Make Graph
    train_op, sum_L2, logit_, loss_ED = make_train_graph(X, Y, is_training)

    ##### Run Session
    sess = tf.Session()

    ##### Check the Checkpoint
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())


    ED = sess.run(loss_ED, feed_dict={X: input_batch,
                                        Y: label_batch,
                                        is_training: False})

    for i in range(len(input_batch)):
        log_txt.write('\n[%d] - %.3f'%(i, ED[i]))

    log_txt.close()
    print('mean loss: %.3f' % np.mean(ED))

def cross_val(max_epoch) :
    ##### Data Loader
    DataLoader=data_load.DataLoader()
    DataLoader.set_subject_path([60, 61, 62, 63, 64, 69, 70, 73, 74, 75])

    ##### training set
    input_batch, label_batch = DataLoader.get_multi_motion([i for i in range(0, 235)], 0, 3, 8)

    #### test set
    test_input_batch, test_label_batch = DataLoader.get_multi_motion([i for i in range(235, 255)], 0, 3)

    ########## don't use because of overflow problem(maybe) ##########
    # ##### Histogram for Visualizing
    # log_ED = {}
    # log_ED['train_ED'] = []
    # log_ED['val_ED'] = []
    ##################################################################

    ##### Loss Log txt file
    log_txt = open('loss.txt', 'a')

    #####Make placeholder
    ### Input
    X = tf.placeholder(tf.float32, [None, 111])
    ### Label
    Y = tf.placeholder(tf.float32, [None, 3])
    ### Is training
    is_training = tf.placeholder(tf.bool)

    ##### Make Graph
    train_op, sum_L2, logit_, loss_ED = make_train_graph(X, Y, is_training)

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
        _, ED = sess.run([train_op, tf.reduce_mean(loss_ED)],
                         feed_dict={X: input_batch, Y: label_batch, is_training: True})

        info_txt = '\n\n[%d/%d] -  mean of train ED: %.3f' % ((epoch + 1), max_epoch, ED)
        log_txt.write(info_txt)
        print(info_txt)
        # log_ED['train_ED'].append(ED) ########## don't use because of overflow problem(maybe) ##########

        if epoch % 1 == 0 :
            ED = sess.run(tf.reduce_mean(loss_ED), feed_dict={X: test_input_batch,
                                                              Y: test_label_batch, is_training: False})

            info_txt = ' /// mean of test ED: %.3f' % ED
            log_txt.write(info_txt)
            print(info_txt)
            # log_ED['val_ED'].append(ED) ########## don't use because of overflow problem(maybe) ##########


    # epoch_ED_plot(log_ED, save=True, path='train_hist.png') ########## don't use because of overflow problem(maybe) ##########
    log_txt.close()
    saver.save(sess, './model/model.ckpt')


cross_val(2000)
# epoch_ED_plot_from_txt('loss.txt')
# inference()
# frame_dist_plot_from_txt('inference_loss.txt')