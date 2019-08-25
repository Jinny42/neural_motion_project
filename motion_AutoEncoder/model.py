import numpy as np
import tensorflow as tf

##########################################################
## INPUT : [Batch, Frame, Joint, Axis, Reference Joint] ##
############# INPUT : [Batch, 64, 11, 3] ##############
############ OUTPUT : [Batch, 64,  4, 3] ##############
##########################################################

def build1(input, is_training, input_size, wd):
    input = tf.reshape(input,[-1, 64, 11, 3])

    resized_input = tf.image.resize_images(input, [64, 16]) # [Batch, 64, 16, 3] : [Batch, Frame, Joint, Axis]

    # H1
    W1 = tf.get_variable("W1", shape=[7, 7, 3, 32], dtype=np.float32,
                          initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(2 / 3)))# He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W1), wd))
    L1 = tf.nn.conv2d(resized_input, W1, strides=[1, 2, 1, 1], padding='SAME')  # [Batch, 32, 16, 32]
    L1 = tf.layers.batch_normalization(L1, training=is_training)
    L1 = tf.nn.relu(L1)

    # H2
    W2 = tf.get_variable("W2", shape=[5, 5, 32, 64], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(2 / 32)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W2), wd))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 2, 1, 1], padding='SAME')  # [Batch, 16, 16, 64]
    L2 = tf.layers.batch_normalization(L2, training=is_training)
    L2 = tf.nn.relu(L2)

    # H3
    W3 = tf.get_variable("W3", shape=[3, 3, 64, 128], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 64)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W3), wd))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 2, 2, 1], padding='SAME')  # [Batch, 8, 8, 128]
    L3 = tf.layers.batch_normalization(L3, training=is_training)
    L3 = tf.nn.relu(L3)

    # H4
    W4 = tf.get_variable("W4", shape=[3, 3, 128, 256], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 128)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W4), wd))
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 2, 2, 1], padding='SAME')  # [Batch, 4, 4, 256]
    L4 = tf.layers.batch_normalization(L4, training=is_training)
    L4 = tf.nn.relu(L4)

    # H5
    W5 = tf.get_variable("W5", shape=[4, 4, 256, 1024], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 256)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W5), wd))
    L5 = tf.nn.conv2d(L4, W5, strides=[1, 2, 2, 1], padding='VALID')  # [Batch, 1, 1, 1024]
    L5 = tf.layers.batch_normalization(L5, training=is_training)
    L5 = tf.nn.relu(L5)

    # 1x1 conv
    W_s = tf.get_variable("11convW1", shape=[1, 1, 1024, 1024], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 1024)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W_s), wd))
    L_s = tf.nn.conv2d(L5, W_s, strides=[1, 1, 1, 1], padding='SAME')  # [Batch, 1, 1, 1024]
    L_s = tf.layers.batch_normalization(L_s, training=is_training)
    L_s = tf.nn.relu(L_s)

    # 1x1 conv
    W_s2 = tf.get_variable("11convW2", shape=[1, 1, 1024, 1024], dtype=np.float32,
                               initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                                   2 / 1024)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W_s2), wd))
    L_s2 = tf.nn.conv2d(L_s, W_s2, strides=[1, 1, 1, 1], padding='SAME')  # [Batch, 1, 1, 1024]
    L_s2 = tf.layers.batch_normalization(L_s2, training=is_training)
    L_s2 = tf.nn.relu(L_s2)

    # U-H1
    W6 = tf.get_variable("W6", shape=[4, 4, 256, 1024], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 1024)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W6), wd))
    L6 = tf.nn.conv2d_transpose(L_s2, W6, output_shape=[input_size, 4, 4, 256],
                                strides=[1, 1, 1, 1], padding='VALID')  # [Batch, 4, 4, 256]
    L6 = tf.layers.batch_normalization(L6, training=is_training)
    L6 = tf.nn.relu(L6)

    # U-H2
    W7 = tf.get_variable("W7", shape=[4, 4, 128, 256], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 256)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W7), wd))
    L7 = tf.nn.conv2d_transpose(L6, W7, output_shape=[input_size, 8, 8, 128],
                                strides=[1, 2, 2, 1], padding='SAME')  # [Batch, 8, 8, 128]
    L7 = tf.layers.batch_normalization(L7, training=is_training)
    L7 = tf.nn.relu(L7)

    # U-H3
    W8 = tf.get_variable("W8", shape=[4, 1, 64, 128], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 128)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W8), wd))
    L8 = tf.nn.conv2d_transpose(L7, W8, output_shape=[input_size, 16, 8, 64],
                                strides=[1, 2, 1, 1], padding='SAME')  # [Batch, 16, 8, 64]
    L8 = tf.layers.batch_normalization(L8, training=is_training)
    L8 = tf.nn.relu(L8)

    # U-H4
    W9 = tf.get_variable("W9", shape=[4, 1, 32, 64], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 64)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W9), wd))
    L9 = tf.nn.conv2d_transpose(L8, W9, output_shape=[input_size, 32, 8, 32],
                                strides=[1, 2, 1, 1], padding='SAME')  # [Batch, 32, 8, 32]
    L9 = tf.layers.batch_normalization(L9, training=is_training)
    L9 = tf.nn.relu(L9)

    # U-H5
    W10 = tf.get_variable("W10", shape=[4, 4, 3, 32], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 32)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W10), wd))
    L10 = tf.nn.conv2d_transpose(L9, W10, output_shape=[input_size, 64, 16, 3],
                                strides=[1, 2, 2, 1], padding='SAME')  # [Batch, 64, 16, 16]

    resized_output = tf.image.resize_images(L10, [64, 11]) # [Batch, 64, 8, 3]


    output1 = resized_output[:, :, 2:4, :]# [Batch, 64, 2, 3]
    output2 = resized_output[:, :, 7:9, :]# [Batch, 64, 2, 3]

    output = tf.concat([output1, output2], axis=2) # [Batch, 64, 4, 3]

    return output #[Batch, 64, 4, 3]


def build2(input, is_training, input_size, wd):
    input = tf.reshape(input,[-1, 64, 11, 3])

    resized_input = tf.image.resize_images(input, [64, 16]) # [Batch, 64, 16, 3] : [Batch, Frame, Joint, Axis]
    # H1
    W1 = tf.get_variable("W1", shape=[7, 7, 3, 32], dtype=np.float32,
                          initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(2 / 3)))# He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W1), wd))
    L1 = tf.nn.conv2d(resized_input, W1, strides=[1, 2, 1, 1], padding='SAME')  # [Batch, 32, 16, 32]
    L1 = tf.layers.batch_normalization(L1, training=is_training)
    L1 = tf.nn.relu(L1)

    # H2
    W2 = tf.get_variable("W2", shape=[5, 5, 32, 64], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(2 / 32)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W2), wd))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 2, 1, 1], padding='SAME')  # [Batch, 16, 16, 64]
    L2 = tf.layers.batch_normalization(L2, training=is_training)
    L2 = tf.nn.relu(L2)

    # H3
    W3 = tf.get_variable("W3", shape=[3, 3, 64, 128], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 64)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W3), wd))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 2, 2, 1], padding='SAME')  # [Batch, 8, 8, 128]
    L3 = tf.layers.batch_normalization(L3, training=is_training)
    L3 = tf.nn.relu(L3)

    # H4
    W4 = tf.get_variable("W4", shape=[3, 3, 128, 256], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 128)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W4), wd))
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 2, 2, 1], padding='SAME')  # [Batch, 4, 4, 256]
    L4 = tf.layers.batch_normalization(L4, training=is_training)
    L4 = tf.nn.relu(L4)

    # H5
    W5 = tf.get_variable("W5", shape=[4, 4, 256, 1024], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 256)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W5), wd))
    L5 = tf.nn.conv2d(L4, W5, strides=[1, 2, 2, 1], padding='VALID')  # [Batch, 1, 1, 1024]
    L5 = tf.layers.batch_normalization(L5, training=is_training)
    L5 = tf.nn.relu(L5)

    # 1x1 conv
    W_s = tf.get_variable("11convW1", shape=[1, 1, 1024, 1024], dtype=np.float32,
                          initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                              2 / 1024)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W_s), wd))
    L_s = tf.nn.conv2d(L5, W_s, strides=[1, 1, 1, 1], padding='SAME')  # [Batch, 1, 1, 1024]
    L_s = tf.layers.batch_normalization(L_s, training=is_training)
    L_s = tf.nn.relu(L_s)

    # 1x1 conv
    W_s2 = tf.get_variable("11convW2", shape=[1, 1, 1024, 1024], dtype=np.float32,
                           initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                               2 / 1024)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W_s2), wd))
    L_s2 = tf.nn.conv2d(L_s, W_s2, strides=[1, 1, 1, 1], padding='SAME')  # [Batch, 1, 1, 1024]
    L_s2 = tf.layers.batch_normalization(L_s2, training=is_training)
    L_s2 = tf.nn.relu(L_s2)

    # U-H1
    W6 = tf.get_variable("W6", shape=[4, 4, 256, 1024], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 1024)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W6), wd))
    L6 = tf.nn.conv2d_transpose(L_s2, W6, output_shape=[input_size, 4, 4, 256],
                                strides=[1, 1, 1, 1], padding='VALID')  # [Batch, 4, 4, 256]
    L6 = tf.layers.batch_normalization(L6, training=is_training)
    L6 = tf.nn.relu(L6)

    # U-H2
    W7 = tf.get_variable("W7", shape=[4, 1, 128, 256], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 256)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W7), wd))
    L7 = tf.nn.conv2d_transpose(L6, W7, output_shape=[input_size, 8, 4, 128],
                                strides=[1, 2, 1, 1], padding='SAME')  # [Batch, 8, 4, 128]
    L7 = tf.layers.batch_normalization(L7, training=is_training)
    L7 = tf.nn.relu(L7)

    # U-H3
    W8 = tf.get_variable("W8", shape=[4, 1, 64, 128], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 128)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W8), wd))
    L8 = tf.nn.conv2d_transpose(L7, W8, output_shape=[input_size, 16, 4, 64],
                                strides=[1, 2, 1, 1], padding='SAME')  # [Batch, 16, 4, 64]
    L8 = tf.layers.batch_normalization(L8, training=is_training)
    L8 = tf.nn.relu(L8)

    # U-H4
    W9 = tf.get_variable("W9", shape=[4, 1, 32, 64], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 64)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W9), wd))
    L9 = tf.nn.conv2d_transpose(L8, W9, output_shape=[input_size, 32, 4, 32],
                                strides=[1, 2, 1, 1], padding='SAME')  # [Batch, 32, 4, 32]
    L9 = tf.layers.batch_normalization(L9, training=is_training)
    L9 = tf.nn.relu(L9)

    # U-H5
    W10 = tf.get_variable("W10", shape=[4, 1, 3, 32], dtype=np.float32,
                         initializer=tf.truncated_normal_initializer(0, stddev=tf.sqrt(
                             2 / 32)))  # He initialization
    tf.add_to_collection('L2_losses', tf.multiply(tf.nn.l2_loss(W10), wd))
    L10 = tf.nn.conv2d_transpose(L9, W10, output_shape=[input_size, 64, 4, 3],
                                strides=[1, 2, 1, 1], padding='SAME')  # [Batch, 64, 4, 3]

    return L10