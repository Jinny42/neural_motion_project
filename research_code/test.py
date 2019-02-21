from plot_functions import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


def make_model1(X, isTrain):
    W1 = tf.get_variable("W1", shape=[108, 6], dtype=np.float32,
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

def make_model3(X, isTrain):
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


    W3 = tf.get_variable("W3", shape=[512, 512], dtype=np.float32,
                         initializer=tf.random_normal_initializer(0, tf.sqrt(2/512)))
    L3 = tf.matmul(L2, W3)
    L3 = tf.layers.batch_normalization(L3, training=isTrain)
    L3 = tf.nn.relu(L3)

    W4 = tf.get_variable("W4", shape=[512, 512], dtype=np.float32,
                         initializer=tf.random_normal_initializer(0, tf.sqrt(2 / 512)))
    L4 = tf.matmul(L3, W4)
    L4 = tf.layers.batch_normalization(L4, training=isTrain)
    L4 = tf.nn.relu(L4)

    W5 = tf.get_variable("W5", shape=[512, 1024], dtype=np.float32,
                         initializer=tf.random_normal_initializer(0, tf.sqrt(2 / 512)))
    L5 = tf.matmul(L4, W5)
    L5 = tf.layers.batch_normalization(L5, training=isTrain)
    L5 = tf.nn.relu(L5)

    W6 = tf.get_variable("W6", shape=[1024, 1024], dtype=np.float32,
                         initializer=tf.random_normal_initializer(0, tf.sqrt(2 / 1024)))
    L6 = tf.matmul(L5, W6)
    L6 = tf.layers.batch_normalization(L6, training=isTrain)
    L6 = tf.nn.relu(L6)

    W7 = tf.get_variable("W7", shape=[1024, 512], dtype=np.float32,
                         initializer=tf.random_normal_initializer(0, tf.sqrt(2 / 1024)))
    L7 = tf.matmul(L6, W7)
    L7 = tf.layers.batch_normalization(L7, training=isTrain)
    L7 = tf.nn.relu(L7)

    W8 = tf.get_variable("W8", shape=[512, 3], dtype=np.float32,
                         initializer=tf.random_normal_initializer(0, tf.sqrt(2 / 512)))
    L8 = tf.matmul(L7, W8)

    return L8

def make_model4(X, isTrain):
    W1 = tf.get_variable("W1", shape=[111, 256], dtype=np.float32,
                                  initializer=tf.random_normal_initializer(0, tf.sqrt(2/111)))# He initialization
    B1 =  tf.get_variable("B1", shape=[256], dtype=np.float32,
                                  initializer=tf.zeros_initializer())
    L1 = tf.matmul(X, W1)
    L1 = tf.nn.bias_add(L1, B1)
    L1 = tf.nn.relu(L1)

    W2 = tf.get_variable("W2", shape=[256, 512], dtype=np.float32,
                         initializer=tf.random_normal_initializer(0, tf.sqrt(2/256)))
    B2 = tf.get_variable("B2", shape=[512], dtype=np.float32,
                         initializer=tf.zeros_initializer())
    L2 = tf.matmul(L1, W2)
    L2 = tf.nn.bias_add(L2, B2)
    L2 = tf.nn.relu(L2)


    W3 = tf.get_variable("W3", shape=[512, 1024], dtype=np.float32,
                         initializer=tf.random_normal_initializer(0, tf.sqrt(2/512)))
    B3 = tf.get_variable("B3", shape=[1024], dtype=np.float32,
                         initializer=tf.zeros_initializer())
    L3 = tf.matmul(L2, W3)
    L3 = tf.nn.bias_add(L3, B3)
    L3 = tf.nn.relu(L3)

    W4 = tf.get_variable("W4", shape=[1024, 3], dtype=np.float32,
                         initializer=tf.random_normal_initializer(0, tf.sqrt(2 / 1024)))
    L4 = tf.matmul(L3, W4)

    return L4

def make_model5(X, isTrain):
    W1 = tf.get_variable("W1", shape=[216, 256], dtype=np.float32,
                         initializer=tf.random_normal_initializer(0, tf.sqrt(2 / 111)))  # He initialization
    B1 = tf.get_variable("B1", shape=[256], dtype=np.float32,
                         initializer=tf.zeros_initializer())
    L1 = tf.matmul(X, W1)
    L1 = tf.nn.bias_add(L1, B1)
    L1 = tf.nn.relu(L1)

    W2 = tf.get_variable("W2", shape=[256, 12], dtype=np.float32,
                         initializer=tf.random_normal_initializer(0, tf.sqrt(2 / 256)))
    L2 = tf.matmul(L1, W2)

    return L2

def make_train_graph(input, label, is_training, gpu_num, split_num):
    input_list = tf.split(input, gpu_num * split_num)
    label_list = tf.split(label, gpu_num * split_num)
    logit_list=[]
    sum_L2_list = []
    ED_list = []
    pck_list = []

    iter = -1

    for d in range(gpu_num):
        with tf.device('/gpu:'+str(d)):
            for i in range(split_num) :
                iter = iter + 1
                with tf.variable_scope(tf.get_variable_scope(), reuse= iter > 0):
                    logit = make_model5(input_list[iter], is_training)
                    loss_L2 = tf.pow(logit - label_list[iter], 2)
                    sum_L2 = tf.reduce_sum(loss_L2)
                    ED = tf.sqrt(tf.reduce_sum(loss_L2, axis=1))
                    pck = tf.to_float(tf.greater(0.3, ED))

                    logit_list.append(logit)
                    sum_L2_list.append(sum_L2)
                    ED_list.append(ED)
                    pck_list.append(pck)

    total_logit = tf.concat(logit_list, axis=0)
    total_sum_L2 = tf.reduce_sum(sum_L2_list)
    total_ED = tf.concat(ED_list, axis=0)
    total_pck = tf.concat(pck_list, axis=0)

    # reduce_mean 대신 reduce_sum을 사용했을때 더 잘 수렴했음
    # batch size와 output node 수에 비례해 scale을 해줬다고 볼 수 있는데
    # facebook에서 발표한 논문에서도 이게 꽤 괜찮은 테크닉이라고 소개하고 있음.
    # 논문링크 : https://research.fb.com/publications/accurate-large-minibatch-sgd-training-imagenet-in-1-hour/
    # 관련링크 : https://stats.stackexchange.com/questions/201452/is-it-common-practice-to-minimize-the-mean-loss-over-the-batches-instead-of-the/201540#201540

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #옵티마이저를 정의할 때 colocate_gradients_with_ops=True 옵션을 주어 Forward가 일어났던 GPU에서 Gradient 계산도 일어나도록 처리.
        #http://openresearch.ai/t/tensorpack-multigpu/45
        train_op = tf.train.AdamOptimizer(0.0002).minimize(sum_L2, colocate_gradients_with_ops=True)

    return train_op, total_sum_L2, total_logit, total_ED, total_pck

def train(input, label, max_epoch, gpu_num, split_num) :
    #####Make placeholder
    ### Input
    X = tf.placeholder(tf.float32, [None, 111])
    ### Label
    Y = tf.placeholder(tf.float32, [None, 3])
    ### Is training
    is_training = tf.placeholder(tf.bool)

    ##### Make Graph
    train_op, sum_L2, logit_, loss_ED, pck = make_train_graph(X, Y, is_training, gpu_num, split_num)
    mean_ED = tf.reduce_mean(loss_ED)

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
        _, ED = sess.run([train_op, mean_ED],
                             feed_dict={X: input, Y: label, is_training: True})

        if epoch +1 % 100 == 0 :
            saver.save(sess, './model/model.ckpt')
            print('saved')

        print('[%d/%d] - mean ED: %.3f'
              % ((epoch + 1), max_epoch, ED))

def inference(gpu_num=2, split_num=1) :
    input_batch = np.load('test_input_batch2.npy')
    label_batch = np.load('test_label_batch2.npy')

    test_len = len(input_batch)

    if test_len % (gpu_num * split_num) != 0 :
        input_batch = input_batch[:test_len // (gpu_num * split_num) *(gpu_num * split_num)]
        label_batch = label_batch[:test_len // (gpu_num * split_num) *(gpu_num * split_num)]

    ##### Loss Log txt file
    log_txt = open('inference_loss.txt', 'a')

    #####Make placeholder
    ### Input
    X = tf.placeholder(tf.float32, [None, 108])
    ### Label
    Y = tf.placeholder(tf.float32, [None, 6])
    ### Is training
    is_training = tf.placeholder(tf.bool)

    ##### Make Graph
    train_op, sum_L2, logit_, loss_ED, pck_ = make_train_graph(X, Y, is_training, gpu_num, split_num)

    ##### Run Session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    ##### Check the Checkpoint
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())


    ED, pck = sess.run([loss_ED, pck_], feed_dict={X: input_batch,
                                        Y: label_batch,
                                        is_training: False})

    for i in range(len(input_batch)):
        log_txt.write('\n[%d] - %.3f / %.3f'%(i, ED[i], pck[i]))

    log_txt.close()
    print('mean ED: %.3f, mean PCK: %.3f' % (np.mean(ED), np.mean(pck)))

def cross_val(max_epoch, gpu_num=2, split_num=1, already_done_epoch=0) :
    input_batch = np.load('input_batch3.npy')
    label_batch = np.load('label_batch3.npy')
    test_input_batch = np.load('test_input_batch3.npy')
    test_label_batch = np.load('test_label_batch3.npy')

    train_len = len(input_batch)
    test_len = len(test_input_batch)

    if train_len % (gpu_num * split_num) != 0 :
        input_batch = input_batch[:train_len // (gpu_num * split_num) *(gpu_num * split_num)]
        label_batch = label_batch[:train_len //(gpu_num * split_num) *(gpu_num * split_num)]

    if test_len % (gpu_num * split_num) != 0 :
        test_input_batch = test_input_batch[:test_len // (gpu_num * split_num) *(gpu_num * split_num)]
        test_label_batch = test_label_batch[:test_len // (gpu_num * split_num) *(gpu_num * split_num)]

    ##### Loss Log txt file
    log_txt = open('loss.txt', 'a')

    #####Make placeholder
    ### Input
    X = tf.placeholder(tf.float32, [None, 216])
    ### Label
    Y = tf.placeholder(tf.float32, [None, 12])
    ### Is training
    is_training = tf.placeholder(tf.bool)

    ##### Make Graph
    train_op, sum_L2, logit_, loss_ED, pck_ = make_train_graph(X, Y, is_training, gpu_num, split_num)
    mean_ED = tf.reduce_mean(loss_ED)
    mean_pck = tf.reduce_mean(pck_)

    ##### Run Session
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    a= time.time()
    sess = tf.Session()
    b=time.time()
    print(b-a)

    ##### Check the Checkpoint
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    start_time = time.time()

    for epoch in range(max_epoch-already_done_epoch):
        ##Update
        _, ED, pck = sess.run([train_op, mean_ED, mean_pck],
                             feed_dict={X: input_batch, Y: label_batch, is_training: True})

        info_txt = '\n[%d/%d] -  mean of train ED: %.3f // train PCK: %.3f' % ((epoch + 1 + already_done_epoch),
                                                                               max_epoch, ED, pck)
        log_txt.write(info_txt)
        print(info_txt)

        ED, pck = sess.run([mean_ED, mean_pck], feed_dict={X: test_input_batch,
                                                          Y: test_label_batch, is_training: False})

        info_txt = ' /// mean of test ED: %.3f //// test PCK: %.3f\n' % (ED, pck)
        log_txt.write(info_txt)
        print(info_txt)
        if epoch % 100 == 0:
            end_time = time.time()
            print (end_time - start_time)
            saver.save(sess, './model/model.ckpt')
            start_time = end_time

    log_txt.close()
    saver.save(sess, './model/model.ckpt')



# total_start_time= time.time()
cross_val(100000, 2, 1, 0)
# print(time.time()-total_start_time)
epoch_ED_plot_from_txt('loss.txt')

# inference()
# frame_dist_plot_from_txt('inference_loss.txt')