# -*- coding: utf-8 -*-
# @Time    : 2018/8/24 17:50
# @Author  : MengnanChen
# @FileName: tacotron.py
# @Software: PyCharm Community Edition

import tensorflow as tf
from tacotron2.utils.symbols import symbols
from tacotron2.models.modules import Prenet,DecoderRNN,FrameProjection,StopProjection,PostNet,post_cbhg
from tacotron2.models.attention import LocationSensitiveAttention
from tacotron2.models.architecture_wrappers import TacotronDecoderCell
from tacotron2.models.helpers import TacoTrainingHelper
from tacotron2.models.custom_decoder import CustomDecoder
from tensorflow.contrib.seq2seq import dynamic_decode
from tacotron2.infolog import log

class Tacotron():
    def __init__(self,hparams):
        self._hparams=hparams

    def initialize(self,inputs,input_lengths,stop_token_targets=None,mel_target=None,
                   linear_targets=None,target_length=None,gta=False,global_step=None,
                   is_training=False,is_evaluating=False):
        '''
        初始化模型
        :param inputs: [batch_size,T_in], T_in是输入时间序列，value: 字符ids
        :param input_lengths: [batch_size], value: 输入时间序列长度
        :param stop_token_targets:
        :param mel_target: [batch_size,T_out,M], 仅在`训练`时使用
        :param linear_targets:
        :param target_length:
        :param gta: ground truth align
        :param global_step:
        :param is_training:
        :param is_evaluating:
        :return:
        '''
        if mel_target is None and stop_token_targets is not None:
            raise ValueError('no mel targets but provide stop_token_targets')
        if mel_target is not None and stop_token_targets is None and not gta:
            raise ValueError('provide mel_target but no corresponding stop_token_targets')
        if not gta and self._hparams.predict_linear==True and linear_targets is None and is_training:
            raise ValueError('model is set to use post-processing to predict linear spectrograms in training but no linear target given')
        if gta and linear_targets is not None:
            raise ValueError('linear spectrogram is not supported in gta mode')
        if is_training and self._hparams.mask_decoder and target_length is not None:
            raise RuntimeError('model set to mask paddings but no targets lengths provided for mask')
        if is_training and is_evaluating:
            raise RuntimeError('model can not be in training and evaluation mode at the same time')

        with tf.variable_scope('inference') as scope:
            batch_size=tf.shape(inputs)[0]
            hp=self._hparams
            assert hp.tacotron_teacher_forcing_mode in ('constant','scheduled')
            if hp.tacotron_teacher_forcing_mode=='scheduled' and is_training:
                assert global_step is not None

            post_condition=hp.predict_linear and not gta

            embedding_table=tf.get_variable(
                'input_embedding',[len(symbols),hp.embedding_dim],dtype=tf.float32
            )
            embedded_inputs=tf.nn.embedding_lookup(embedding_table,inputs)

            encoder_cell=TacotronEncoderCell(
                EncoderConvolutions(is_training,hparams=hp,scope='encoder_convolution'),
                EncoderRNN(is_training,size=hp.encoder_lstm_units,
                zoneout=hp.tacotron_zoneout_rate,scope='encoder_LSTM')
            )
            encoder_outputs=encoder_cell(embedded_inputs,input_lengths)

            enc_conv_output_shape=encoder_cell.conv_output_shape

            prenet=Prenet(is_training,layers_sizes=hp.prenet_layers,drop_rate=hp.tacotron_dropout_rate,
                          scope='decoder_prenet')

            attention_mechanism=LocationSensitiveAttention(hp.attention_dim,
                                                           encoder_outputs,
                                                           mask_encoder=hp.mask_encoder,
                                                           memory_sequence_length=input_lengths,
                                                           smoothing=hp.smoothing,
                                                           cumulate_weights=hp.cumulative_weights)
            decoder_lstm=DecoderRNN(is_training,layers=hp.decoder_layers,
                                    size=hp.decoder_lstm_units,
                                    zoneout=hp.tacotron_zoneout_rate,
                                    scope='decoder_lstm')
            frame_projection=FrameProjection(hp.num_mels*hp.output_per_step,scope='linear_transform')
            stop_projection=StopProjection(is_training or is_evaluating,shape=hp.output_per_step,scope='stop_token_projection')

            decoder_cell=TacotronDecoderCell(
                prenet,
                attention_mechanism,
                decoder_lstm,
                frame_projection,
                stop_projection
            )

            if is_training or is_evaluating or gta:
                self.helper=TacoTrainingHelper(batch_size,mel_target,stop_token_targets,
                                               hp,gta,is_evaluating,global_step)
            else:
                self.helper=TacoTrainingHelper(batch_size,hp)

            decoder_init_state=decoder_cell.zero_state(batch_size=batch_size,dtype=tf.float32)
            # Only use max iterations at synthesis time
            max_iters=hp.max_iters if not (is_training or is_evaluating) else None
            (frame_projection,stop_token_prediction,_),final_decoder_state,_=dynamic_decode(
                CustomDecoder(decoder_cell,self.helper,decoder_init_state),
                impute_finished=False,
                maximum_iterations=max_iters,
                swap_memory=hp.tacotron_swap_with_cpu
            )

            decoder_output=tf.reshape(frame_projection,[batch_size,-1,hp.num_mels])
            stop_token_prediction=tf.reshape(stop_token_prediction,[batch_size,-1])

            # [batch_size, decoder_steps*r, postnet_channels]
            postnet=PostNet(is_training,hparams=hp,scope='postnet_convolutions')
            residual=postnet(decoder_output)

            # [batch_size, decoder_steps*r, num_mels]
            residual_projection=FrameProjection(hp.num_mels,scope='postnet_projection')
            projected_residual=residual_projection(residual)

            mel_outputs=decoder_output+projected_residual  # residual

            if post_condition:
                post_outputs=post_cbhg(mel_outputs,hp.num_mels,is_training)
                linear_outputs=tf.layers.dense(post_outputs,hp.num_freqs)

            alignments=tf.transpose(final_decoder_state.alignment_history.stack(),[1,2,0])
            if is_training:
                self.ratio=self.helper._ratio
            self.inputs = inputs
            self.input_lengths = input_lengths
            self.decoder_output = decoder_output
            self.alignments = alignments
            self.stop_token_prediction = stop_token_prediction
            self.stop_token_targets = stop_token_targets
            self.mel_outputs = mel_outputs
            if post_condition:
                self.linear_outputs=linear_outputs
                self.linear_targets=linear_targets
            self.mel_targets=mel_target
            self.targets_lengths=target_length
            log('Initialized Tacotron model. Dimension (? = dynamic shape):')
            log('Train mode:{}'.format(is_training))
            log('Eval mode:{}'.format(is_evaluating))
            log('GTA mode:{}'.format(gta))
            log('Synthesis mode:{}'.format(gta))
            log('embedding:{}'.format(embedded_inputs.shape))
            log('enc conv out:{}'.format(enc_conv_output_shape))
            log('encoder out:{}'.format(encoder_outputs.shape))
            log('decoder out:{}'.format(decoder_output.shape))
            log('residual out:{}'.format(residual.shape))
            log('projected residual out:{}'.format(projected_residual.shape))
            log('mel out:{}'.format(mel_outputs.shape))
            if post_condition:
                log('linear out:{}'.format(linear_outputs.shape))
            log('<stop token> out:{}'.format(stop_token_prediction.shape))

    def add_loss(self):
        with tf.variable_scope('loss') as scope:
            hp=self._hparams

            if hp.mask_encoder:
                pass
                # before postnet