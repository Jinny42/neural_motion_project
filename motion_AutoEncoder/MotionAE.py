import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import model
import dataset
import os

class MotionAE:
    def __init__(self):
        self.model_kind = None
        self.input_size = None
        self.lr = None
        self.wd = None
        self.momentum = None
        self.done_epoch = None
        # self.loss_sampling_step = 10

        self.graph = None


    def make_train_op(self, dataset_loader, is_training, gpu_num, split_num):
        mean_L2_list = []
        input_list = []
        label_list = []
        logit_list = []
        iter_ = -1

        for d in range(gpu_num):
            with tf.device('/gpu:' + str(d)):
                for i in range(split_num):
                    iter_ = iter_ + 1
                    with tf.variable_scope(tf.get_variable_scope(), reuse=iter_ > 0):
                        input, label = dataset_loader.get_next()
                        if self.model_kind == 1:
                            logit = model.build1(input, is_training, self.input_size, self.wd)
                        elif self.model_kind == 2:
                            logit = model.build2(input, is_training, self.input_size, self.wd)
                        elif self.model_kind == 3:
                            logit = model.build3(input, is_training, self.input_size, self.wd)
                            label = tf.expand_dims(
                                tf.expand_dims(label[:, -1, 0, :], axis=1),
                                                   axis=1)  # [Batch, 1, 1, 3]
                        # else:
                        #     logit = model.build4(input, is_training)

                        L2_loss = tf.pow(logit - label, 2)
                        mean_L2 = tf.reduce_mean(L2_loss)

                        input_list.append(input)
                        label_list.append(label)
                        logit_list.append(logit)
                        mean_L2_list.append(mean_L2)

        total_input = tf.concat(input_list, axis=0)
        total_label = tf.concat(label_list, axis=0)
        total_logit = tf.concat(logit_list, axis=0)
        total_sum_L2 = tf.reduce_mean(mean_L2_list)
        tf.add_to_collection('losses', total_sum_L2)

        total_loss = tf.add_n(tf.get_collection('losses'))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(total_loss, colocate_gradients_with_ops=True)
            # train_op = tf.train.MomentumOptimizer(self.lr, self.momentum).minimize(total_sum_L2, colocate_gradients_with_ops=True)

        ###### L2-loss가 더해진, Training Operation에 사용되는 진짜 loss는
        ###### 패러미터 수에 따라 그 값이 커지기 때문에 모델간 성능 비교를 위해선 loss대신 CEE(Cross Entropy Error)를 반환한다.
        return train_op, total_sum_L2, total_input, total_label, total_logit


    def train(self, max_epoch, model_kind, input_size, lr, wd, momentum, done_epoch, gpu_num=2, split_num=1):
        self.model_kind = model_kind
        self.input_size = input_size
        self.lr = lr
        self.wd = wd
        self.momentum = momentum
        self.done_epoch = done_epoch

        self.graph = tf.Graph()

        with self.graph.as_default():
            ###data 로드
            dataset_loader = dataset.Dataset(input_size)
            train_init_op, test_init_op = dataset_loader.get_initializer_for_training()
            is_training = tf.placeholder(tf.bool)

            ##### Loss Log txt file
            log_txt = open('loss.txt', 'a')

            ##### Make Graph
            train_op, loss_ , input_, label_, logit_ = self.make_train_op(dataset_loader, is_training, gpu_num, split_num)

            ##### Run Session
            # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            sess = tf.Session()


            ##### Check the Checkpoint
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state('./model_ckpt')
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())



            for epoch in range(max_epoch - self.done_epoch):
                train_loss_list = []
                test_loss_list = []

                start_time = time.time()

                ##Update
                sess.run(train_init_op)

                iter_per_epoch = (dataset_loader.train_len) // (input_size * gpu_num * split_num)
                for iter in range(iter_per_epoch):
                    _, loss, logit, label = sess.run([train_op, loss_, logit_, label_],
                                       feed_dict={is_training: True})
                    # print(loss)
                    train_loss_list.append(loss)


                info_txt = '\n[%d/%d] -  mean of train L2: %.3f'%(epoch + 1 + self.done_epoch,
                                                                  max_epoch, np.mean(train_loss_list))

                log_txt.write(info_txt)
                print(info_txt)


                ##Validation
                sess.run(test_init_op)

                iter_per_epoch = (dataset_loader.total_len - dataset_loader.train_len)//(input_size * gpu_num * split_num)
                for iter in range(iter_per_epoch):
                    loss = sess.run(loss_, feed_dict={is_training: False})
                    test_loss_list.append(loss)

                info_txt = ' // mean of test L2: %.3f' % (np.mean(test_loss_list))

                log_txt.write(info_txt)
                print(info_txt)

                ptime_per_epoch = time.time()- start_time
                info_txt = ' /// ptime: %.3f\n' % (ptime_per_epoch)
                log_txt.write(info_txt)
                print(info_txt)

                if (epoch+1) % 10 == 0:
                    saver.save(sess, './model_ckpt/model.ckpt')


            log_txt.close()
            saver.save(sess, './model_ckpt/model.ckpt')


    def inference(self, model_kind, input_size, lr, wd, momentum, input_path, gpu_num=2, split_num=1):
        self.model_kind = model_kind
        self.input_size = input_size
        self.lr = lr
        self.wd = wd
        self.momentum = momentum

        self.graph = tf.Graph()

        with self.graph.as_default():
            ###data 로드
            dataset_loader = dataset.Dataset(input_size, root_path = input_path)
            inference_init_op = dataset_loader.get_initializer_for_inference()
            is_training = tf.placeholder(tf.bool)

            ##### Make Graph
            train_op, loss_, input_, label_, logit_ = self.make_train_op(dataset_loader, is_training, gpu_num, split_num)

            ##### Run Session
            # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            sess = tf.Session()


            ##### Check the Checkpoint
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state('./model_ckpt')
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            inference_loss_list = []
            result_list = []

            sess.run(inference_init_op)

            iter_per_epoch = (dataset_loader.total_len) // (input_size * gpu_num * split_num) + 1
            for iter in range(iter_per_epoch):
                loss, logit = sess.run([loss_, logit_], feed_dict={is_training: False})
                inference_loss_list.append(loss)

                if(self.model_kind == 1 or self.model_kind == 2) : # logit : [Batch, 64, 4, 3]
                    logit = logit[:, -1, :, :] # [Batch, 4, 3]
                result_list.append(logit)

            info_txt = '\nL2 Loss: %.3f'%(np.mean(inference_loss_list))
            print(info_txt)

            output = np.concatenate(result_list, axis=0) # [Batch * N, 4, 3]
            np.save(os.path.join(input_path ,'output.npy'), output) #[Frame - 63, 2, 3, 3]

    def make_train_op2(self, input, label, is_training, gpu_num, split_num):
        mean_L2_list = []
        input_list = tf.split(input, gpu_num * split_num)
        label_list = tf.split(label, gpu_num * split_num)
        logit_list = []
        iter_ = -1

        for d in range(gpu_num):
            with tf.device('/gpu:' + str(d)):
                for i in range(split_num):
                    iter_ = iter_ + 1
                    with tf.variable_scope(tf.get_variable_scope(), reuse=iter_ > 0):
                        if self.model_kind == 1:
                            logit = model.build1(input_list[iter_], is_training, self.input_size // (gpu_num * split_num), self.wd)
                            label = label_list[iter_]
                        elif self.model_kind == 2:
                            logit = model.build2(input_list[iter_], is_training, self.input_size // (gpu_num * split_num), self.wd)
                            label = label_list[iter_]
                        elif self.model_kind == 3:
                            logit = model.build3(input_list[iter_], is_training, self.input_size // (gpu_num * split_num), self.wd)
                            label = tf.expand_dims(
                                tf.expand_dims(label_list[iter_][:, -1, 0, :], axis=1),
                                                   axis=1)  # [Batch, 1, 1, 3]
                        # else:
                        #     logit = model.build4(input, is_training)

                        L2_loss = tf.pow(logit - label, 2)
                        mean_L2 = tf.reduce_mean(L2_loss)
                        logit_list.append(logit)
                        mean_L2_list.append(mean_L2)

        total_logit = tf.concat(logit_list, axis=0)
        total_sum_L2 = tf.reduce_mean(mean_L2_list)
        tf.add_to_collection('losses', total_sum_L2)

        total_loss = tf.add_n(tf.get_collection('losses'))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(total_loss, colocate_gradients_with_ops=True)
            # train_op = tf.train.MomentumOptimizer(self.lr, self.momentum).minimize(total_sum_L2, colocate_gradients_with_ops=True)

        ###### L2-loss가 더해진, Training Operation에 사용되는 진짜 loss는
        ###### 패러미터 수에 따라 그 값이 커지기 때문에 모델간 성능 비교를 위해선 loss대신 CEE(Cross Entropy Error)를 반환한다.
        return train_op, total_sum_L2, total_logit

    def inference2(self, model_kind, input_size, lr, wd, momentum, input_path, gpu_num=2, split_num=1):
        self.model_kind = model_kind
        self.input_size = input_size
        self.lr = lr
        self.wd = wd
        self.momentum = momentum

        self.graph = tf.Graph()

        with self.graph.as_default():
            ###data 로드
            X = tf.placeholder(tf.float32, [None, 64, 11, 3])
            Y = tf.placeholder(tf.float32, [None, 64, 4, 3])
            is_training = tf.placeholder(tf.bool)

            ##### Make Graph
            train_op, loss_, logit_ = self.make_train_op2(X, Y, is_training, gpu_num, split_num)

            ##### Run Session
            # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            sess = tf.Session()


            ##### Check the Checkpoint
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state('./model_ckpt')
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            inference_loss_list = []
            result_list = []

            total_path_list = []
            file_names = os.listdir(input_path)
            for file_name in file_names:
                total_path_list.append(os.path.join(input_path, file_name))
            total_len = len(total_path_list)

            iter_per_epoch = (total_len // input_size) + 1
            for iter in range(iter_per_epoch):
                input_batch = np.zeros([input_size, 64, 11, 3], dtype=np.float32)
                label_batch = np.zeros([input_size, 64, 4, 3], dtype=np.float32)

                for i in range(input_size):
                    idx = (iter * input_size) + i
                    if idx < total_len :
                        input_batch[i] = np.load(total_path_list[idx])
                    else:
                        input_batch[i] = np.zeros([64, 11, 3])

                for num, idx in enumerate((2, 3, 7, 8)):
                    label_batch[:, :, num, :] = input_batch[:, :, idx, :]
                    input_batch[:, :, idx, :] = 0
                # input_batch[:, :, 1, :] = 0
                # input_batch[:, :, 5, :] = 0
                # input_batch[:, :, 9, :] = 0

                loss, logit = sess.run([loss_, logit_], feed_dict={X: input_batch, Y:label_batch, is_training: False})
                inference_loss_list.append(loss)

                if(self.model_kind == 1 or self.model_kind == 2) : # logit : [Batch, 64, 4, 3]
                    logit = logit[:, -1, :, :] # [Batch, 4, 3]
                result_list.append(logit)

            info_txt = '\nL2 Loss: %.3f'%(np.mean(inference_loss_list))
            print(info_txt)

            output = np.concatenate(result_list, axis=0) # [Batch * N, 4, 3]
            np.save(os.path.join(input_path ,'output.npy'), output) #[Frame - 63, 2, 3, 3]

