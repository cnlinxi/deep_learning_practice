# -*- coding: utf-8 -*-
# @Time    : 2018/10/18 10:52
# @Author  : MengnanChen
# @FileName: inception.py
# @Software: PyCharm

import tensorflow as tf


slim=tf.contrib.slim
trunc_normal=lambda stddev:tf.truncated_normal(0.,stddev=stddev)


def inception_v3_arg_scope(weight_decay=4e-4,
                           stddev=1e-1,
                           batch_norm_var_collection='moving_vars'):
    batch_norm_params={
        'decay':0.9997,
        'epsilon':1e-2,
        'updates_collections':tf.GraphKeys.UPDATE_OPS,
        'variables_collections':{
            'beta':None,
            'gamma':None,
            'moving_mean':[batch_norm_var_collection],
            'moving_variance':[batch_norm_var_collection],
        }
    }
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params
        ) as sc:
            return sc


def inception_v3_base(inputs,scope=None):
    end_points={}

    with tf.variable_scope(scope,'InceptionV3',[inputs]):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='VALID'):
            net=slim.conv2d(inputs,32,[3,3],stride=2,scope='conv2d_1a_3x3')
            net=slim.conv2d(net,32,[3,3],scope='conv2d_2a_3x3')
            net=slim.conv2d(net,64,[3,3],scope='conv2d_2b_3x3')
            net=slim.max_pool2d(net,[3,3],stride=2,scope='maxpool_3a_3x3')
            net=slim.conv2d(net,80,[1,1],scope='conv2d_3b_1x1')
            net=slim.conv2d(net,192,[3,3],scope='conv2d_4a_3x3')
            net=slim.max_pool2d(net,[3,3],stride=2,scope='maxpool_5a_3x3')

        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='SAME'):
            with tf.variable_scope('mixed_5b'):
                with tf.variable_scope('branch_0'):
                    branch_0=slim.conv2d(net,64,[1,1],scope='conv2d_0a_1x1')
                with tf.variable_scope('branch_1'):
                    branch_1=slim.conv2d(net,48,[1,1],scope='conv2d_0a_1x1')
                    branch_1=slim.conv2d(branch_1,64,[5,5],scope='conv2d_0b_5x5')
                with tf.variable_scope('branch_2'):
                    branch_2=slim.conv2d(net,64,[1,1],scope='conv2d_0a_1x1')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='conv2d_0b_3x3')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='conv2d_0c_3x3')
                with tf.variable_scope('branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='avgpool_0a_3x3')
                    branch_3=slim.conv2d(branch_3,32,[1,1],scope='conv2d_0b_1x1')
                net=tf.concat([branch_0,branch_1,branch_2,branch_3],axis=-1)

            with tf.variable_scope('mixed_5c'):
                with tf.variable_scope('branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='conv2d_b0_1x1')
                with tf.variable_scope('branch_1'):
                    branch_1=slim.conv2d(net,48,[1,1],scope='conv2d_b1_1_1x1')
                    branch_1=slim.conv2d(branch_1,64,[5,5],scope='conv2d_b1_2_5x5')
                with tf.variable_scope('branch_2'):
                    branch_2=slim.conv2d(net,64,[1,1],scope='conv2d_b2_1_1x1')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='conv2d_b2_2_3x3')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='conv2d_b2_3_3x3')
                with tf.variable_scope('branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='avgpool_b3_1_3x3')
                    branch_3=slim.conv2d(branch_3,64,[1,1],scope='conv2d_b3_2_3x3')
                net=tf.concat([branch_0,branch_1,branch_2,branch_3],axis=-1)

            with tf.variable_scope('mixed_5d'):
                with tf.variable_scope('branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='conv2d_b0_1x1')
                with tf.variable_scope('branch_1'):
                    branch_1=slim.conv2d(net,48,[1,1],scope='conv2d_b1_1_1x1')
                    branch_1=slim.conv2d(branch_1,64,[5,5],scope='conv2d_b1_2_5x5')
                with tf.variable_scope('branch_2'):
                    branch_2=slim.conv2d(net,64,[1,1],scope='conv2d_b2_1_1x1')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='conv2d_b2_2_3x3')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='conv2d_b2_3_3x3')
                with tf.variable_scope('branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='avgpool_b3_1_3x3')
                    branch_3=slim.conv2d(branch_3,64,[1,1],scope='conv2d_b3_2_3x3')
                net=tf.concat([branch_0,branch_1,branch_2,branch_3],axis=-1)

            with tf.variable_scope('mixed_6a'):
                with tf.variable_scope('branch_0'):
                    branch_0=slim.conv2d(net,384,[3,3],scope='conv2d_b0_1_3x3')
                with tf.variable_scope('branch_1'):
                    branch_1=slim.conv2d(net,64,[1,1],scope='conv2d_b1_1_1x1')
                    branch_1=slim.conv2d(branch_1,96,[3,3],scope='conv2d_b1_2_3x3')
                    branch_1=slim.conv2d(branch_1,96,[3,3],stride=2,padding='VALID',scope='conv2d_b1_3_3x3')
                with tf.variable_scope('branch_2'):
                    branch_2=slim.max_pool2d(net,[3,3],stride=2,padding='VALID',scope='maxpool_b2_1_3x3')
                net=tf.concat([branch_0,branch_1,branch_2],axis=-1)

            with tf.variable_scope('mixed_6b'):
                with tf.variable_scope('branch_0'):
                    branch_0=slim.conv2d(net,192,[1,1],scope='conv2d_b0_1_1x1')
                with tf.variable_scope('branch_1'):
                    branch_1=slim.conv2d(net,128,[1,1],scope='conv2d_b1_1_1x1')
                    branch_1=slim.conv2d(branch_1,128,[1,7],scope='conv2d_b1_2_1x7')
                    branch_1=slim.conv2d(branch_1,192,[7,1],scope='conv2d_b1_3_7x1')
                with tf.variable_scope('branch_2'):
                    branch_2=slim.conv2d(net,128,[1,1],scope='conv2d_b2_1_1x1')
                    branch_2=slim.conv2d(branch_2,128,[7,1],scope='conv2d_b2_2_7x1')
                    branch_2=slim.conv2d(branch_2,128,[1,7],scope='conv2d_b2_3_1x7')
                    branch_2=slim.conv2d(branch_2,128,[7,1],scope='conv2d_b2_4_7x1')
                    branch_2=slim.conv2d(branch_2,192,[1,7],scope='conv2d_b2_5_1x7')
                with tf.variable_scope('branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='avgpool_b3_1_3x3')
                    branch_3=slim.conv2d(branch_3,192,[1,1],scope='conv2d_b3_2_1x1')
                net=tf.concat([branch_0,branch_1,branch_2,branch_3],axis=-1)

            with tf.variable_scope('mixed_6c'):
                with tf.variable_scope('branch_0'):
                    branch_0=slim.conv2d(net,192,[1,1],scope='conv2d_b0_1_1x1')
                with tf.variable_scope('branch_1'):
                    branch_1=slim.conv2d(net,160,[1,1],scope='conv2d_b1_1_1x1')
                    branch_1=slim.conv2d(branch_1,[1,7],scope='conv2d_b1_2_1x7')
                    branch_1=slim.conv2d(branch_1,[7,1],scope='conv2d_b1_3_7x1')
                with tf.variable_scope('branch_2'):
                    branch_2=slim.conv2d(net,160,[1,1],scope='conv2d_b2_1_1x1')
                    branch_2=slim.conv2d(branch_2,160,[7,1],scope='conv2d_b2_2_7x1')
                    branch_2=slim.conv2d(branch_2,160,[1,7],scope='conv2d_b2_3_1x7')
                    branch_2=slim.conv2d(branch_2,160,[7,1],scope='conv2d_b2_4_7x1')
                    branch_2=slim.conv2d(branch_2,192,[1,7],scope='conv2d_b2_5_1x7')
                with tf.variable_scope('branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='avgpool_b3_1_3x3')
                    branch_3=slim.conv2d(branch_3,192,[1,1],scope='conv2d_b3_2_1x1')
                net=tf.concat([branch_0,branch_1,branch_2,branch_3],axis=-1)

            with tf.variable_scope('mixed_6d'):
                with tf.variable_scope('branch_0'):
                    branch_0=slim.conv2d(net,192,[1,1],scope='conv2d_b0_1_1x1')
                with tf.variable_scope('branch_1'):
                    branch_1=slim.conv2d(net,160,[1,1],scope='conv2d_b1_1_1x1')
                    branch_1=slim.conv2d(branch_1,[1,7],scope='conv2d_b1_2_1x7')
                    branch_1=slim.conv2d(branch_1,[7,1],scope='conv2d_b1_3_7x1')
                with tf.variable_scope('branch_2'):
                    branch_2=slim.conv2d(net,160,[1,1],scope='conv2d_b2_1_1x1')
                    branch_2=slim.conv2d(branch_2,160,[7,1],scope='conv2d_b2_2_7x1')
                    branch_2=slim.conv2d(branch_2,160,[1,7],scope='conv2d_b2_3_1x7')
                    branch_2=slim.conv2d(branch_2,160,[7,1],scope='conv2d_b2_4_7x1')
                    branch_2=slim.conv2d(branch_2,192,[1,7],scope='conv2d_b2_5_1x7')
                with tf.variable_scope('branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='avgpool_b3_1_3x3')
                    branch_3=slim.conv2d(branch_3,192,[1,1],scope='conv2d_b3_2_1x1')
                net=tf.concat([branch_0,branch_1,branch_2,branch_3],axis=-1)

            with tf.variable_scope('mixed_6e'):
                with tf.variable_scope('branch_0'):
                    branch_0=slim.conv2d(net,192,[1,1],scope='conv2d_b0_1_1x1')
                with tf.variable_scope('branch_1'):
                    branch_1=slim.conv2d(net,192,[1,1],scope='conv2d_b1_1_1x1')
                    branch_1=slim.conv2d(branch_1,[1,7],scope='conv2d_b1_2_1x7')
                    branch_1=slim.conv2d(branch_1,[7,1],scope='conv2d_b1_3_7x1')
                with tf.variable_scope('branch_2'):
                    branch_2=slim.conv2d(net,192,[1,1],scope='conv2d_b2_1_1x1')
                    branch_2=slim.conv2d(branch_2,192,[7,1],scope='conv2d_b2_2_7x1')
                    branch_2=slim.conv2d(branch_2,192,[1,7],scope='conv2d_b2_3_1x7')
                    branch_2=slim.conv2d(branch_2,192,[7,1],scope='conv2d_b2_4_7x1')
                    branch_2=slim.conv2d(branch_2,192,[1,7],scope='conv2d_b2_5_1x7')
                with tf.variable_scope('branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='avgpool_b3_1_3x3')
                    branch_3=slim.conv2d(branch_3,192,[1,1],scope='conv2d_b3_2_1x1')
                net=tf.concat([branch_0,branch_1,branch_2,branch_3],axis=-1)
            end_points['mixed_6e']=net

            with tf.variable_scope('mixed_7a'):
                with tf.variable_scope('branch_0'):
                    branch_0=slim.conv2d(net,192,[1,1],scope='conv2d_b0_1_1x1')
                    branch_0=slim.conv2d(branch_0,320,[3,3],scope='conv2d_b0_2_1x1')
                with tf.variable_scope('branch_1'):
                    branch_1=slim.conv2d(net,192,[1,1],scope='conv2d_b1_1_1x1')
                    branch_1=slim.conv2d(branch_1,192,[1,7],scope='conv2d_b1_2_1x7')
                    branch_1=slim.conv2d(branch_1,192,[7,1],scope='conv2d_b1_3_7x1')
                    branch_1=slim.conv2d(branch_1,192,[3,3],stride=2,padding='VALID',scope='conv2d_b1_4_3x3')
                with tf.variable_scope('branch_2'):
                    branch_2=slim.max_pool2d(net,[3,3],stride=2,padding='VALID',scope='maxpool_b2_1_3x3')
                net=tf.concat([branch_0,branch_1,branch_2],axis=-1)

            with tf.variable_scope('mixed_7b'):
                with tf.variable_scope('branch_0'):
                    branch_0=slim.conv2d(net,320,[1,1],scope='conv2d_b0_1_1x1')
                with tf.variable_scope('branch_1'):
                    branch_1=slim.conv2d(net,384,[1,1],scope='conv2d_b1_1_1x1')
                    branch_1=tf.concat([
                        slim.conv2d(branch_1,384,[1,3],scope='conv2d_b1_2_1x3'),
                        slim.conv2d(branch_1,384,[3,1],scope='conv2d_b1_3_3x1')
                    ],axis=-1)
                with tf.variable_scope('branch_2'):
                    branch_2=slim.conv2d(net,448,[1,1],scope='conv2d_b2_1_1x1')
                    branch_2=slim.conv2d(branch_2,384,[3,3],scope='conv2d_b2_2_3x3')
                    branch_2=tf.concat([
                        slim.conv2d(branch_2,384,[1,3],scope='conv2d_b2_3_1x3'),
                        slim.conv2d(branch_2,384,[3,1],scope='conv2d_b2_4_3x1')
                    ],axis=-1)
                with tf.variable_scope('branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='avgpool_b3_1_3x3')
                    branch_3=slim.conv2d(branch_3,192,[1,1],scope='conv2d_b3_1x1')
                net=tf.concat([branch_0,branch_1,branch_2,branch_3],axis=-1)

            with tf.variable_scope('mixed_7c'):
                with tf.variable_scope('branch_0'):
                    branch_0=slim.conv2d(net,320,[1,1],scope='conv2d_b0_1_1x1')
                with tf.variable_scope('branch_1'):
                    branch_1=slim.conv2d(net,384,[1,1],scope='conv2d_b1_1_1x1')
                    branch_1=tf.concat([
                        slim.conv2d(branch_1,384,[1,3],scope='conv2d_b1_2_1x3'),
                        slim.conv2d(branch_1,384,[3,1],scope='conv2d_b1_3_3x1')
                    ],axis=-1)
                with tf.variable_scope('branch_2'):
                    branch_2=slim.conv2d(net,448,[1,1],scope='conv2d_b2_1_1x1')
                    branch_2=slim.conv2d(branch_2,384,[3,3],scope='conv2d_b2_2_3x3')
                    branch_2=tf.concat([
                        slim.conv2d(branch_2,384,[1,3],scope='conv2d_b2_3_1x3'),
                        slim.conv2d(branch_2,384,[3,1],scope='conv2d_b2_4_3x1')
                    ],axis=-1)
                with tf.variable_scope('branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='avgpool_b3_1_3x3')
                    branch_3=slim.conv2d(branch_3,192,[1,1],scope='conv2d_b3_1x1')
                net=tf.concat([branch_0,branch_1,branch_2,branch_3],axis=-1)

    return net,end_points


def inception_v3(inputs,
                 num_classes=1000,
                 is_training=True,
                 drop_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV3'):
    with tf.variable_scope(scope,'InceptionV3',[inputs,num_classes],reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm,slim.dropout],is_training=is_training):
            net,end_points=inception_v3_base(inputs,scope=scope)

    with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                        stride=1,padding='SAME'):
        aux_logits=end_points['mixed_6e']

        with tf.variable_scope('auxlogits'):
            aux_logits=slim.avg_pool2d(aux_logits,[5,5],stride=3,padding='VALID',scope='avgpool_aux_1_5x5')
            aux_logits=slim.conv2d(aux_logits,128,[1,1],scope='conv2d_aux_2_1x1')
            aux_logits=slim.conv2d(aux_logits,768,[5,5],
                                   weights_initializer=trunc_normal(1e-2),
                                   padding='VALID',
                                   scope='conv2d_aux_3_5x5')
            aux_logits=slim.conv2d(aux_logits,num_classes,[1,1],activation_fn=None,
                                   normalizer_fn=None,weight_initializer=trunc_normal(1e-3),
                                   scope='conv2d_aux_4_1x1')
            if spatial_squeeze:
                aux_logits=tf.squeeze(aux_logits,axis=[1,2],name='aux_squeeze')
            end_points['aux_squeeze']=aux_logits

        with tf.variable_scope('logits'):
            net=slim.avg_pool2d(net,[8,8],padding='VALID',scope='avgpool_aux_1_8x8')
            net=slim.dropout(net,keep_prob=drop_keep_prob,scope='dropout_aux_2')
            end_points['prelogits']=net
            logits=slim.conv2d(net,num_classes,[1,1],activation_fn=None,normalizer_fn=None,
                               scope='conv2d_aux_3_1x1')
            if spatial_squeeze:
                logits=tf.squeeze(logits,axis=[1,2],name='logit_squeeze')
            end_points['logits']=logits
            end_points['predictions']=prediction_fn(logits,scope='predictions')
