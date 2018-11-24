# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 17:01
# @Author  : MengnanChen
# @FileName: feeder.py
# @Software: PyCharm

import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf

from . import text
from . import hparams as hp


class Feeder:
    def __init__(self, data_paths, labels, sess):
        super(Feeder, self).__init__()
        self._data_paths = data_paths
        self._labels = labels
        self._sess = sess

        text_preprocessor = text.TextProcess(hp.w2v_model_path)
        self._pad = text_preprocessor.un_index
        self.vocab_size = text_preprocessor.vocab_size
        self.w2v_table=text_preprocessor.w2v_array
        seq_lengths = []
        datas = None
        ground_truths = None
        for data_path, label in zip(self._data_paths, labels):
            datas_per_file = []
            with open(data_path, 'rb') as fin:
                for line in fin:
                    line = line.decode('utf-8').strip('\r\n ')
                    line = text_preprocessor.text_to_sequence(line)  # [1,5,7,..]
                    seq_lengths.append(len(line))
                    datas_per_file.append(line)
            if label==1:
                ground_truths_per_file = [label for _ in range(len(datas_per_file))]
            else:
                ground_truths_per_file=[label for _ in range(len(datas_per_file))]
            datas = np.concatenate((datas, datas_per_file)) if datas is not None else datas_per_file
            ground_truths = np.concatenate(
                (ground_truths, ground_truths_per_file)) if ground_truths is not None else ground_truths_per_file

        self.max_seq_len = max(seq_lengths)
        datas=np.asarray(datas)
        seq_lengths=np.asarray(seq_lengths)
        ground_truths=np.asarray(ground_truths)

        def _pad_input(x, length):
            return np.pad(x, (0, length - len(x)), mode='constant', constant_values=self._pad)

        input_xs = []
        for sentence in datas:
            input_xs.append(_pad_input(sentence, self.max_seq_len))
        # input_xs: [batch_size,max_seq_len]
        # input_lengths: [batch_size,]
        # input_ys: [batch_size,]
        shuffle(input_xs, seq_lengths, ground_truths)

        eval_input_xs = input_xs[:hp.test_size]
        eval_input_lengths = seq_lengths[:hp.test_size]
        eval_input_ys = ground_truths[:hp.test_size]

        input_xs = input_xs[hp.test_size:]
        input_lengths = seq_lengths[hp.test_size:]
        input_ys = ground_truths[hp.test_size:]

        self.num_batch = len(input_xs) // hp.batch_size
        self.eval_num_batch = len(eval_input_xs) // hp.batch_size

        self._placeholder = (
            tf.placeholder(tf.int32, shape=(None, self.max_seq_len), name='input_x'),  # [batch_size,seq_len]
            tf.placeholder(tf.int32, shape=(None,), name='input_lengths'),  # [batch_size,]
            tf.placeholder(tf.int32, shape=(None,), name='input_y')  # [batch_size,]
        )

        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_tensor_slices(self._placeholder)
            dataset = dataset.batch(hp.batch_size).shuffle(buffer_size=1000).repeat()
            iterator = dataset.make_initializable_iterator()
            self._sess.run(iterator.initializer,
                           feed_dict=dict(zip(self._placeholder, (input_xs, input_lengths, input_ys))))
            self.next_element = iterator.get_next()

            eval_dataset = tf.data.Dataset.from_tensor_slices(self._placeholder)
            eval_dataset = eval_dataset.batch(hp.batch_size).repeat()
            eval_iterator = eval_dataset.make_initializable_iterator()
            self._sess.run(eval_iterator.initializer,
                           feed_dict=dict(zip(self._placeholder, (eval_input_xs, eval_input_lengths, eval_input_ys))))
            self.eval_next_element = eval_iterator.get_next()
