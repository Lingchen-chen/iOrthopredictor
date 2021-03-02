from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def vgg_19(inputs, reuse=False, pooling='max', final_endpoint='fc8'):
    def conv2d(input_,
               kernel_size,
               stride,
               num_outputs,
               scope,
               activation_fn=tf.nn.relu):
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size is expected to be odd.')
        padding = kernel_size // 2
        padded_input = tf.pad(input_,
                              [[0, 0], [padding, padding], [padding, padding], [0, 0]],
                              mode='REFLECT')
        return slim.conv2d(padded_input,
                           padding='VALID',
                           kernel_size=kernel_size,
                           stride=stride,
                           num_outputs=num_outputs,
                           activation_fn=activation_fn,
                           scope=scope)

    pooling_fns = {'avg': slim.avg_pool2d, 'max': slim.max_pool2d}
    pooling_fn = pooling_fns[pooling]
    with tf.variable_scope('vgg_19', [inputs], reuse=tf.AUTO_REUSE):
        # cause the default value of reuse in the slim functions or the customs' functions
        # is None, then its value depends on the high level's reuse value
        end_points = {}

        def add_and_check_is_final(layer_name, net):
            end_points['%s' % (layer_name)] = net
            return layer_name == final_endpoint

        with slim.arg_scope([slim.conv2d], trainable=False):
            # slim.conv2d's defalut stride is 1 ; 2 is the number of the repetition
            # the parameters initialized in slim.repeate are named as
            # respectively 'vgg19/conv1/conv1_1'  and 'vgg19/conv1/conv1_2'
            with tf.variable_scope('preprocess'):
                net = conv2d(inputs, kernel_size=1, stride=1, num_outputs=3, scope='conv1', activation_fn=None)

            with tf.variable_scope('relu1'):
                net = conv2d(net, kernel_size=3, stride=1, num_outputs=64, scope='relu1_1')
                if add_and_check_is_final('relu1_1', net): return end_points

                net = conv2d(net, kernel_size=3, stride=1, num_outputs=64, scope='relu1_2')
                if add_and_check_is_final('relu1_2', net): return end_points

            net = pooling_fn(net, [2, 2], scope='pool1')
            if add_and_check_is_final('pool1', net): return end_points

            with tf.variable_scope('relu2'):
                net = conv2d(net, kernel_size=3, stride=1, num_outputs=128, scope='relu2_1')
                if add_and_check_is_final('relu2_1', net): return end_points

                net = conv2d(net, kernel_size=3, stride=1, num_outputs=128, scope='relu2_2')
                if add_and_check_is_final('relu2_2', net): return end_points

            net = pooling_fn(net, [2, 2], scope='pool2')
            if add_and_check_is_final('pool2', net): return end_points

            with tf.variable_scope('relu3'):
                net = conv2d(net, kernel_size=3, stride=1, num_outputs=256, scope='relu3_1')
                if add_and_check_is_final('relu3_1', net): return end_points

                net = conv2d(net, kernel_size=3, stride=1, num_outputs=256, scope='relu3_2')
                if add_and_check_is_final('relu3_2', net): return end_points

                net = conv2d(net, kernel_size=3, stride=1, num_outputs=256, scope='relu3_3')
                if add_and_check_is_final('relu3_3', net): return end_points

                net = conv2d(net, kernel_size=3, stride=1, num_outputs=256, scope='relu3_4')
                if add_and_check_is_final('relu3_4', net): return end_points

            net = pooling_fn(net, [2, 2], scope='pool3')
            if add_and_check_is_final('pool3', net): return end_points

            with tf.variable_scope('relu4'):

                net = conv2d(net, kernel_size=3, stride=1, num_outputs=512, scope='relu4_1')
                if add_and_check_is_final('relu4_1', net): return end_points

                net = conv2d(net, kernel_size=3, stride=1, num_outputs=512, scope='relu4_2')
                if add_and_check_is_final('relu4_2', net): return end_points

                net = conv2d(net, kernel_size=3, stride=1, num_outputs=512, scope='relu4_3')
                if add_and_check_is_final('relu4_3', net): return end_points

                net = conv2d(net, kernel_size=3, stride=1, num_outputs=512, scope='relu4_4')
                if add_and_check_is_final('relu4_4', net): return end_points

            net = pooling_fn(net, [2, 2], scope='pool4')
            if add_and_check_is_final('pool4', net): return end_points

            with tf.variable_scope('relu5'):

                net = conv2d(net, kernel_size=3, stride=1, num_outputs=512, scope='relu5_1')
                if add_and_check_is_final('relu5_1', net): return end_points

                net = conv2d(net, kernel_size=3, stride=1, num_outputs=512, scope='relu5_2')
                if add_and_check_is_final('relu5_2', net): return end_points

                net = conv2d(net, kernel_size=3, stride=1, num_outputs=512, scope='relu5_3')
                if add_and_check_is_final('relu5_3', net): return end_points

                net = conv2d(net, kernel_size=3, stride=1, num_outputs=512, scope='relu5_4')
                if add_and_check_is_final('relu5_4', net): return end_points

            net = pooling_fn(net, [2, 2], scope='pool5')
            if add_and_check_is_final('pool5', net): return end_points

        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)


