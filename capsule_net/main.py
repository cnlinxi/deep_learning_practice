#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/9 16:42
# @Author  : MengnanChen
# @Site    : 
# @File    : main.py
# @Software: PyCharm Community Edition

import os
from config import cfg
from utils import load_data
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from capsule_net import capsule_net

def save_to():
    if not os.path.exists(cfg.result):
        os.mkdir(cfg.result)
    if cfg.is_training:
        loss='{}/loss.csv'.format(cfg.result)
        train_acc='{}/train_acc.csv'.format(cfg.result)
        val_acc='{}/val_acc.csv'.format(cfg.result)

        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)
        if os.path.exists(val_acc):
            os.remove(val_acc)

        fp_loss=open(loss,'w')
        fp_loss.write('step,loss\n')
        fp_train_acc=open(train_acc,'w')
        fp_train_acc.write('step,train_acc\n')
        fp_val_acc=open(val_acc,'w')
        fp_val_acc.write('step,val_acc\n')

        return fp_train_acc,fp_val_acc,fp_loss
    else:
        test_acc='{}/test_acc.csv'.format(cfg.result)
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fp_test_acc=open(test_acc,'w')
        fp_test_acc.write('step,test_acc\n')
        return fp_test_acc

def train(model, supervisor, num_label):
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data(cfg.dataset, cfg.batch_size,is_train=True)
    Y = valY[:num_val_batch * cfg.batch_size].reshape((-1, 1))

    fd_train_acc, fd_loss, fd_val_acc = save_to()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with supervisor.managed_session(config=config) as sess:
        print("\nNote: all of results will be saved to directory: " + cfg.result)
        for epoch in range(cfg.epoch):
            print("Training for epoch %d/%d:" % (epoch, cfg.epoch))
            if supervisor.should_stop():
                print('supervisor stoped!')
                break
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                global_step = epoch * num_tr_batch + step

                if global_step % cfg.train_sum_freq == 0:
                    _, loss, train_acc, summary_str = sess.run([model.train_op, model.total_loss, model.accuracy, model.train_summary])
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()
                else:
                    sess.run(model.train_op)

                if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:
                    val_acc = 0
                    for i in range(num_val_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        acc = sess.run(model.accuracy, {model.X: valX[start:end], model.labels: valY[start:end]})
                        val_acc += acc
                    val_acc = val_acc / (cfg.batch_size * num_val_batch)
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()

            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()


def evaluation(model, supervisor, num_label):
    teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)
    fd_test_acc = save_to()
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')

        test_acc = 0
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size
            acc = sess.run(model.accuracy, {model.X: teX[start:end], model.labels: teY[start:end]})
            test_acc += acc
        test_acc = test_acc / (cfg.batch_size * num_te_batch)
        fd_test_acc.write(str(test_acc))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + cfg.results + '/test_acc.csv')

def main(_):
    tf.logging.info('Loading Graph...')
    num_labels=10
    model=capsule_net()
    tf.logging.info('Graph loaded')

    sv=tf.train.Supervisor(graph=model.graph,logdir=cfg.logdir,save_model_secs=0)
    if cfg.is_training:
        tf.logging.info('Start training...')
        train(model,sv,num_labels)
        tf.logging.info('training done')
    else:
        evaluation(model,sv,num_labels)

if __name__ == '__main__':
    tf.app.run()