# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 16:27
# @Author  : MengnanChen
# @FileName: hparams.py
# @Software: PyCharm

w2v_model_path = 'data/sgns.financial.bigram-char'
data_paths=['data/neg.txt','data/pos.txt']
labels=[0,1]
embedding_size = 300  # pretrained word2vec model embedding size
embedding_resize = 256

n_classes = 2

test_size = 128
batch_size = 2

transformer_hidden_units = 256
transformer_num_blocks = 4
transformer_num_heads = 8

restore=True
drop_rate = 0.5
reg_weight = 1e-3  # l2 regularization weight
initial_learning_rate = 1e-3  # initial learning rate
decay_rate = 0.96
decay_steps = 2000
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-6
clip_gradients = True

train_steps = 10000
summary_interval=1000
eval_interval=1000
checkpoint_interval=1000
