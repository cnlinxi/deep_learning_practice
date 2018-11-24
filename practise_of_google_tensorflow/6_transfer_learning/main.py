# -*- coding: utf-8 -*-
# @Time    : 2018/10/7 20:06
# @Author  : MengnanChen
# @FileName: main.py
# @Software: PyCharm

import os
import glob
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# Inception-v3 瓶颈节点变量名
BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

STEPS = int(1e4)
LEARNING_RATE = 1e-3
BATCH_SIZE = 4

# 下载完成的模型存储位置
MODEL_DIR = 'model/'
MODEL_FILE = 'classify_image.pb'

# 经瓶颈层特征抽取获得的特征向量临时保存的位置
CACHE_DIR = 'tmp/'

INPUT_DATA = 'input/'

# 测试集和验证集的比例
TEST_PERCENTAGE = 10
VALIDATION_PERCENTAGE = 10


# 生成数据列表
def create_image_lists(testing_percentage, validation_percentage):
    result = {}
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    # 默认os.walk主要有目录，一直会向下遍历，直至没有目录为止，每组返回[root:str,dirs:list,files:list]
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # 只取最顶层的目录所对应的[root,dirs,files]
    # bottleneck vector缓存目录
    bottleneck_vector_dir = os.path.join(INPUT_DATA, 'bottleneck_vector')
    os.makedirs(bottleneck_vector_dir, exist_ok=True)
    file_to_vec = {}
    label_list = []
    for sub_dir in sub_dirs[1]:
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(sub_dirs[0], sub_dir, '*.{}'.format(extension))
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue
        label_name = dir_name.lower()
        label_list.extend(label_name)
        training_images = []
        testing_images = []
        validation_images = []
        for image_file_dir in file_list:
            file_basename = os.path.basename(image_file_dir)
            file_to_vec[image_file_dir] = os.path.join(bottleneck_vector_dir,
                                                       '{}_{}.vec'.format(label_name, file_basename))
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(image_file_dir)
            elif chance < (validation_percentage + testing_percentage):
                testing_images.append(image_file_dir)
            else:
                training_images.append(image_file_dir)
        result[label_name] = {
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
        }
    result['file_to_vec'] = file_to_vec
    result['label_list'] = label_list
    return result


# 使用模型处理图片，获取其在瓶颈处的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 经过瓶颈层的卷积神经网络处理得到的是四维张量，需要将其压缩为一维张量
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


# 获取image_dir瓶颈处特征向量
def get_or_create_bottleneck(sess, image_dir, image_metadata, jpeg_data_tensor, bottleneck_tensor):
    file_to_vec_dict = image_metadata['file_to_vec']
    image_bottleneck_dir = file_to_vec_dict[image_dir]
    # 如果没有生成的bottleneck文件
    if not os.path.exists(image_bottleneck_dir):
        image_data = gfile.FastGFile(image_dir, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(image_bottleneck_dir, 'wb') as fin:
            fin.write(bottleneck_string.encode())
    else:
        with open(image_bottleneck_dir, 'rb') as fout:
            bottleneck_string = fout.read()
        bottleneck_values = [float(x) for x in bottleneck_string.decode().split(',')]
    return bottleneck_values


# 获取training/testing/validation，大小为batch_size的数据集
def get_random_cached_bottlenecks(sess, image_metadata, batch_size, jpeg_data_tensor, bottleneck_tensor,
                                  dataset_mode='training'):
    bottlenecks = []
    ground_truths = []
    label_list = image_metadata['label_list']
    n_classes = len(label_list)
    if dataset_mode == 'testing':
        for label in label_list:
            for image_dir in image_metadata[label]['testing']:
                bottleneck_vec = get_or_create_bottleneck(sess, image_dir, image_metadata, jpeg_data_tensor,
                                                          bottleneck_tensor)
                ground_truth = np.zeros(n_classes, dtype=np.float32)
                ground_truth[label_list.index(label)] = 1.
                bottlenecks.append(bottleneck_vec)
                ground_truths.append(ground_truth)
    else:
        for _ in range(batch_size):
            label_index = random.randint(0, n_classes - 1)
            label = label_list[label_index]
            image_dir = random.sample(image_metadata[label][dataset_mode], 1)[0]
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.
            bottleneck_vec = get_or_create_bottleneck(sess, image_dir, image_metadata, jpeg_data_tensor,
                                                      bottleneck_tensor)
            bottlenecks.append(bottleneck_vec)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def main():
    image_metadata = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_metadata['label_list'])
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as fin:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fin.read())

        # 加载模型，并返回数据输入对应的张量和瓶颈层所对应的张量
        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def,
                                                                  return_elements=[BOTTLENECK_TENSOR_NAME,
                                                                                   JPEG_DATA_TENSOR_NAME])
        bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='bottleneck_input')
        ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='ground_truth_input')

        with tf.name_scope('final_training_ops'):
            weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
            biases = tf.Variable(tf.zeros([n_classes]))
            logits = tf.matmul(bottleneck_input, weights) + biases
            final_tensor = tf.nn.softmax(logits)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, ground_truth_input)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

        with tf.name_scope('evalation'):
            correct_prediction = tf.equal(tf.argmax(final_tensor, axis=1), tf.argmax(ground_truth_input, axis=1))
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(train_step)

            for i in range(STEPS):
                train_bottlenecks, train_ground_truth = \
                    get_random_cached_bottlenecks(sess, image_metadata, BATCH_SIZE, jpeg_data_tensor, bottleneck_tensor)
                sess.run(train_step,
                         feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

                if i % int(1e2) == 0 or i + 1 == STEPS:
                    validation_bottlenecks, validation_ground_truth = \
                        get_random_cached_bottlenecks(sess, image_metadata, BATCH_SIZE, jpeg_data_tensor,
                                                      bottleneck_tensor,
                                                      dataset_mode='validation')
                    validation_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: validation_bottlenecks,
                                                                               ground_truth_input: validation_ground_truth})
                    print('step {}: Validation accuracy on random sampled {} example={}'.format(i, BATCH_SIZE,
                                                                                                validation_accuracy * 100))

            test_bottlenecks, test_ground_truth = get_random_cached_bottlenecks(sess, image_metadata, None,
                                                                                jpeg_data_tensor, bottleneck_tensor,
                                                                                dataset_mode='testing')
            test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks,
                                                                 ground_truth_input: test_ground_truth})
            print('final test accuracy = {}'.format(test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()
