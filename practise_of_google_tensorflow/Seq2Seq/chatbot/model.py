# -*- coding: utf-8 -*-
# @Time    : 2018/10/22 15:53
# @Author  : MengnanChen
# @FileName: model.py
# @Software: PyCharm
# refer to: https://github.com/princewen/tensorflow_practice/blob/master/nlp/chat_bot_seq2seq_attention/model.py

import tensorflow as tf


class Seq2SeqModel():
    def __init__(self,rnn_size,num_layers,embedding_size,learning_rate,
                 word_to_idx,vocab_size,mode,use_attention,beam_search,
                 beam_size,max_gradient_norm=5.):
        self.learning_rate=learning_rate
        self.embedding_size=embedding_size
        self.rnn_size=rnn_size
        self.num_layers=num_layers
        self.word_to_idx=word_to_idx
        self.vocab_size=vocab_size
        self.mode=mode
        self.use_attention=use_attention
        self.beam_search=beam_search
        self.beam_size=beam_size
        self.max_gradient_norm=max_gradient_norm
        self.build_model()

    def _create_rnn_cell(self):
        def single_rnn_cell():
            single_cell=tf.nn.rnn_cell.LSTMCell(self.rnn_size)
            cell=tf.nn.rnn_cell.DropoutWrapper(single_cell,
                                               output_keep_prob=self.keep_prob)
            return cell

        cell=tf.nn.rnn_cell.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def build_model(self):
        print('build model...')
        # placeholder
        self.encoder_inputs=tf.placeholder(tf.int32,[None,None],name='encoder_input')
        self.encoder_inputs_length=tf.placeholder(tf.int32,[None],name='encoder_input_length')

        self.batch_size=tf.placeholder(tf.int32,[],name='batch_size')
        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')

        self.decoder_targets=tf.placeholder(tf.int32,[None,None],name='decoder_targets')
        self.decoder_targets_length=tf.placeholder(tf.int32,[None],name='decoder_targets_length')
        self.max_target_sequence_length=tf.reduce_max(self.decoder_targets_length,name='max_target_len')
        # sequence_mask 第一个参数为序列实际长度，第二个参数为填充长度（一般为序列最大长度），返回True/False张量
        # refer to: https://www.w3cschool.cn/tensorflow_python/tensorflow_python-2elp2jns.html
        self.mask=tf.sequence_mask(self.decoder_targets_length,self.max_target_sequence_length,dtype=tf.int32,name='masks')

        with tf.variable_scope('encoder'):
            encoder_cell=self._create_rnn_cell()
            embedding=tf.get_variable('embedding',[self.vocab_size,self.embedding_size])
            encoder_inputs_emb=tf.nn.embedding_lookup(embedding,self.encoder_inputs)
            # dynamic_rnn构建LSTM模型，将输入编码到隐状态
            # encoder_outputs: batch_size*encoder_inputs_length*rnn_size, 用作attention的memory
            # encoder_state: batch_size*rnn_size, 用作decoder的初始状态
            encoder_outputs,encoder_state=tf.nn.dynamic_rnn(encoder_cell,encoder_inputs_emb,
                                                            sequence_length=self.encoder_inputs_length,
                                                            dtype=tf.float32)

        with tf.variable_scope('decoder'):
            encoder_inputs_length=self.encoder_inputs_length
            attention_mechanism=tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size,
                                                                     memory=encoder_outputs,
                                                                     memory_sequence_length=encoder_inputs_length)
            decoder_cell=self._create_rnn_cell()
            decoder_cell=tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                             attention_mechanism=attention_mechanism,
                                                             attention_layer_size=self.rnn_size,
                                                             name='attention_wrapper')

            decoder_initial_state=decoder_cell.zero_state(batch_size=self.batch_size,
                                                          dtype=tf.float32).clone(cell_state=encoder_state)
            output_layer=tf.layers.Dense(self.vocab_size,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            if self.mode=='train':
                ending=tf.strided_slice(self.decoder_targets,
                                        [0,0],
                                        [self.batch_size,-1],
                                        [1,1])
                decoder_input=tf.concat([tf.fill([self.batch_size,1],self.word_to_idx['<go>']),ending],1)
                decoder_inputs_embedded=tf.nn.embedding_lookup(embedding,decoder_input)
                # embedding -> TrainingHelper -> BasicDecoder -> dynamic_decode -> loss
                # 定义training_helper, 输入为decoder_inputs & sequence_length
                training_helper=tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                  sequence_length=self.decoder_targets_length,
                                                                  time_major=False,
                                                                  name='training_helper')
                # 定义BasicDecoder, 输入为decoder_cell & training_helper & decoder_initial_state & dense_layer
                training_decoder=tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                                 helper=training_helper,
                                                                 initial_state=decoder_initial_state,
                                                                 output_layer=output_layer)
                # 调用dynamic_decoder解码，decoder_outputs是namedtuple，里面包含(rnn_outputs,sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算
                # sample_id: [batch_size]，tf.int32，保存最终的编码结果。可以表示最后的答案
                decoder_outputs,_,_=tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                      impute_finished=True,
                                                                      maximum_iterations=self.max_target_sequence_length)
                self.decoder_logits_train=tf.identity(decoder_outputs.rnn_output)
                self.decoder_predict_train=tf.argmax(self.decoder_logits_train,
                                                     axis=-1,
                                                     name='decoder_predict_train')
                self.loss=tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                           targets=self.decoder_targets_length,
                                                           weights=self.mask)
                tf.summary.scalar('loss',self.loss)
                self.summary_op=tf.summary.merge_all()

                optimizer=tf.train.AdamOptimizer(self.learning_rate)
                trainable_params=tf.trainable_variables()
                gradients=tf.gradients(self.loss,trainable_params)
                clip_gradients,_=tf.clip_by_global_norm(gradients,self.max_gradient_norm)
                self.train_op=optimizer.apply_gradients(zip(clip_gradients,trainable_params))

            elif self.mode=='decode':
                start_token=tf.ones([self.batch_size,],tf.int32)*self.word_to_idx['<go>']
                end_token=self.word_to_idx['<eos>']
                if self.beam_search:
                    inference_decoder=tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                                           embedding=embedding,
                                                                           start_token=start_token,
                                                                           end_token=end_token,
                                                                           initial_state=decoder_initial_state,
                                                                           output_layer=output_layer)
                else:
                    decoding_helper=tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                             start_token=start_token,
                                                                             end_token=end_token)
                    inference_decoder=tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                                      helper=decoding_helper,
                                                                      initial_state=decoder_initial_state,
                                                                      output_layer=output_layer)

                decoder_outputs,_,_=tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                      max_iterations=10)
                # 只是在inference阶段进行beam_search
                # beam_search时，dynamic_decode返回包含两项(predicted_ids,beam_search_decoder_output)
                # predicted_ids: [batch_size,decoder_targets_length,beam_size]，保存输出结果
                # beam_search_decoder_output: BeamSearchDecoderOutput namestuple(score,predicted_ids,parent_ids)
                if self.beam_search:
                    self.decoder_predict_decode=decoder_outputs.predicted_ids
                # 不使用beam_search时，dynamic_decode返回包含两项(rnn_outputs,sample_id)
                # rnn_outputs: [batch_size, decoder_targets_length, vocab_size]
                # sample_id: [batch_size,decoder_targets_length] ,tf.int32
                else:
                    self.decoder_predict_decode=tf.expand_dims(decoder_outputs.sample_id,axis=-1)
        # 保存模型
        self.saver=tf.train.Saver(tf.global_variables())

    def train(self,sess,batch):
        # train阶段，执行self.train_op, self.loss, self.summary_op
        feed_dict={
            self.encoder_inputs:batch.encoder_inputs,
            self.encoder_inputs_length:batch.encoder_inputs_length,
            self.decoder_targets:batch.decoder_targets,
            self.decoder_targets_length:batch.decoder_targets_length,
            self.keep_prob:0.5,
            self.batch_size:len(batch.encoder_inputs)
        }
        _,loss,summary=sess.run([self.train_op,self.loss,self.summary_op],feed_dict=feed_dict)
        return loss,summary

    def eval(self,sess,batch):
        # eval阶段，不要反向传播，只执行self.loss, self.summary_op
        feed_dict={
            self.encoder_inputs:batch.encoder_inputs,
            self.encoder_inputs_length:batch.encoder_inputs_length,
            self.decoder_targets:batch.decoder_targets,
            self.decoder_targets_length:batch.decoder_targets_length,
            self.keep_prob:1.,
            self.batch_size:len(batch.encoder_inputs)
        }
        loss,summary=sess.run([self.loss,self.summary_op],feed_dict=feed_dict)
        return loss,summary

    def infer(self,sess,batch):
        # infer阶段，只需要计算前向，不需要summary_op
        feed_dict={
            self.encoder_inputs:batch.encoder_inputs,
            self.encoder_inputs_length:batch.encoder_inputs_length,
            self.keep_prob:1.,
            self.batch_size:len(batch.encoder_inputs)
        }
        predict=sess.run([self.decoder_predict_decode],feed_dict=feed_dict)
        return predict






