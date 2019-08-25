import tensorflow as tf
import numpy as np
import os
import random

class Dataset :
    def __init__(self, batch_size, root_path= 'preprocessed_data/clips/raw_pos_clips', target_idx=(2, 3, 7, 8)):
        self.total_path_list = []
        file_names = os.listdir(root_path)
        for file_name in file_names:
            self.total_path_list.append(os.path.join(root_path, file_name))
        self.batch_size = batch_size
        self.total_len = len(self.total_path_list) #61138
        self.train_len = 54124
        self.target_idx = target_idx

        self.iterator = None

    def rotation_augmentation(self, dataset):
        random_rotation_angle = tf.random.uniform([1], maxval=np.pi * 2)
        rotation_matrix = tf.squeeze(tf.transpose(tf.stack(
            [
                (tf.cos(random_rotation_angle), tf.constant(0, shape=[1], dtype=tf.float32),
                 tf.sin(random_rotation_angle)),
                (tf.constant(0, shape=[1], dtype=tf.float32), tf.constant(1, shape=[1], dtype=tf.float32),
                 tf.constant(0, shape=[1], dtype=tf.float32)),
                (-tf.sin(random_rotation_angle), tf.constant(0, shape=[1], dtype=tf.float32),
                 tf.cos(random_rotation_angle))
            ], axis=0)
        ))

        dataset = dataset.map(lambda x: tf.reshape(tf.matmul(tf.reshape(x, [self.batch_size * 64 * 11, 3]), rotation_matrix), [self.batch_size, 64, 11, 3]))

        return dataset

    def get_initializer_for_training(self):
        ###set training dataset
        dataset_train = tf.data.Dataset.from_tensor_slices((self.total_path_list[:self.train_len]))
        dataset_train = dataset_train.repeat().shuffle(self.train_len).batch(self.batch_size)
        dataset_train = dataset_train.map(
            lambda x: tf.py_func(self.read_npy_file, [x], [np.float32]))  # [batch_size, 64, 11, 3]


        dataset_train = self.rotation_augmentation(dataset_train)

        dataset_train = dataset_train.map(
            lambda x: tf.py_func(self.preprocess, [x], [tf.float32, tf.float32]))


        ###set test dataset
        dataset_test = tf.data.Dataset.from_tensor_slices((self.total_path_list[self.train_len:]))
        dataset_test = dataset_test.repeat().batch(self.batch_size)
        dataset_test = dataset_test.map(
            lambda x: tf.py_func(self.read_npy_file, [x], [np.float32]))  # [batch_size, 64, 11, 3]
        dataset_test = dataset_test.map(
            lambda x: tf.py_func(self.preprocess, [x], [tf.float32, tf.float32]))

        self.iterator = tf.data.Iterator.from_structure(dataset_train.output_types,
                                                   dataset_train.output_shapes)
        training_init_op = self.iterator.make_initializer(dataset_train)
        validation_init_op = self.iterator.make_initializer(dataset_test)

        return training_init_op, validation_init_op

    def get_initializer_for_inference(self):
        ###set dataset
        dataset_inference = tf.data.Dataset.from_tensor_slices((self.total_path_list))
        dataset_inference = dataset_inference.repeat().batch(self.batch_size)
        dataset_inference = dataset_inference.map(
            lambda x: tf.py_func(self.read_npy_file, [x], [np.float32]))  # [batch_size, 64, 11, 3]
        dataset_inference = dataset_inference.map(
            lambda x: tf.py_func(self.preprocess, [x], [tf.float32, tf.float32]))

        self.iterator = tf.data.Iterator.from_structure(dataset_inference.output_types,
                                                        dataset_inference.output_shapes)
        inference_init_op = self.iterator.make_initializer(dataset_inference)

        return inference_init_op


    def read_npy_file(self, path_list):
        path_len = len(path_list)
        npy_batch = np.empty([path_len, 64, 11, 3], dtype=np.float32)
        for i in range(path_len) :
            npy_batch[i] = np.load(path_list[i].decode())
        return npy_batch

    def preprocess(self, input_batch):
        batch_len = len(input_batch)
        label_batch = np.empty([batch_len, 64, 4, 3], dtype=np.float32)
        for b in range(batch_len) :
            for num, idx in enumerate(self.target_idx):
                label_batch[b, :, num, :] = input_batch[b, :, idx, :]
                input_batch[b, :, idx, :] = 0

        return input_batch, label_batch  # [batch_size, 64, 11, 3], [batch_size, 64, 4, 3]

    def preprocess2(self, input_batch):
        batch_len = len(input_batch)
        label_batch = np.empty([batch_len, 64, 4, 3], dtype=np.float32)
        for b in range(batch_len):
            for num, idx in enumerate(self.target_idx):
                label_batch[b, :, num, :] = input_batch[b, :, idx, :]
                input_batch[b, :, idx, :] = 0
            input_batch[b, :, 1, :] = 0
            input_batch[b, :, 5, :] = 0
            input_batch[b, :, 9, :] = 0

        return input_batch, label_batch # [batch_size, 64, 11, 3], [batch_size, 64, 4, 3]

    def get_next(self):
        return self.iterator.get_next()