# -*- coding: utf-8 -*-
# @Time    : 2018/11/21 9:43
# @Author  : MengnanChen
# @FileName: feeder.py
# @Software: PyCharm

import time
import linecache
import threading

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

### this is text reader, you can use this read big data
# this is no relation with this feeder :)
class CachedLineList:
    def __init__(self, fname):
        self._fname = fname

    def __getitem__(self, x):
        if type(x) is slice:
            return [linecache.getline(self._fname, n + 1)
                    for n in range(x.start, x.stop, x.step)]
        elif type(x) is list or type(x) is np.ndarray:
            return [linecache.getline(self._fname, i)
                    for i in x]
        else:
            return linecache.getline(self._fname, x + 1)

    def __getslice__(self, beg, end):
        return self[beg:end:1]
###


class Feeder:
    def __init__(self, data_path, coordinator, sess):
        super(Feeder, self).__init__()

        self._batch_size = 2
        _n_features = 6553  # the number of feature in .csv file
        self._data_path = data_path
        self._coord = coordinator
        self._sess = sess
        data = pd.read_csv(self._data_path)

        label_name = 'emolabel'
        data[label_name] = LabelEncoder().fit_transform(data[label_name])
        feature_names = [x for x in data.columns if x != label_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data[feature_names].values,
                                                                                data[label_name].values,
                                                                                test_size=0.1)
        self.train_data_index = 0
        self.train_data_length = len(self.X_train)
        self.eval_num_batch = len(self.X_test) // self._batch_size

        with tf.device('/cpu:0'):
            self._placeholder = (
                tf.placeholder(tf.float32, shape=(None, _n_features), name='input_x'),
                tf.placeholder(tf.int32, shape=(None,), name='input_y')
            )

            # train
            queue = tf.FIFOQueue(
                capacity=8,
                dtypes=(tf.float32, tf.int32),
                name='input_queue'
            )
            self._enqueue_op = queue.enqueue(vals=self._placeholder)
            self.inputs, self.labels = queue.dequeue()
            self.inputs.set_shape(self._placeholder[0].shape)
            self.labels.set_shape(self._placeholder[1].shape)

            # eval
            eval_queue = tf.FIFOQueue(
                capacity=1,
                dtypes=(tf.float32, tf.int32),
                name='eval_queue'
            )
            self._eval_enqueue_op = eval_queue.enqueue(vals=self._placeholder)
            self.eval_inputs, self.eval_labels = eval_queue.dequeue()
            self.eval_inputs.set_shape(self._placeholder[0].shape)
            self.eval_labels.set_shape(self._placeholder[1].shape)

    def _get_next_example(self):
        '''
        you can prepare one example in dataset here
        if you are on CV, you should decode and resize image here
        if you are on audio, you should padding data...
        '''
        if self.train_data_index >= self.train_data_length:
            self.train_data_index = 0
        next_example = self.X_train[self.train_data_index]
        self.train_data_index += 1
        return next_example

    def _enqueue_next_group(self):
        # generate n groups batch and you will shuffle data in this n groups
        _batches_per_groups = 4
        while not self._coord.should_stop():
            start_time = time.time()
            examples = [self._get_next_example() for _ in range(self._batch_size * _batches_per_groups)]
            batches = [examples[i:i + self._batch_size] for i in range(0, len(examples), self._batch_size)]
            np.random.shuffle(batches)
            print('generate {} train batches of size {} in {:.3f} sec'.format(_batches_per_groups, self._batch_size,
                                                                              time.time() - start_time))
            # push data into placeholder
            for batch in batches:
                feed_dict = dict(zip(self._placeholder, batch))
                self._sess.run(self._enqueue_op, feed_dict=feed_dict)

    def _enqueue_next_test_group(self):
        # you can prepare test data here
        # grasp all test examples and make them into batches: [batch1, batch2,...]
        # batch1: [example1, example2,...]
        # the followed will make some example never be evaluated
        test_batches = [self.X_test[i * self._batch_size:(i + 1) * self._batch_size] for i in
                        range(0, self.eval_num_batch)]
        while not self._coord.should_stop():
            # push data into placeholder
            for batch in test_batches:
                feed_dict=dict(zip(self._placeholder,batch))
                self._sess.run(self._eval_enqueue_op,feed_dict=feed_dict)

    def start_threads(self):
        # train data input thread
        thread=threading.Thread(name='background',target=self._enqueue_next_group)
        thread.daemon=True
        thread.start()

        # eval data input thread
        thread=threading.Thread(name='background',target=self._enqueue_next_test_group)
        thread.daemon=True
        thread.start()
