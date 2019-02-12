# -*- coding: utf-8 -*-
# @Time    : 2019/2/1 18:13
# @Author  : MengnanChen
# @FileName: tacotron.py
# @Software: PyCharm

import torch
from torch import nn
from .attention import AttentionRNNCell


class Prenet(nn.Module):
    def __init__(self, in_features, out_features=(256, 128)):
        super(Prenet, self).__init__()
        in_features = [in_features] + out_features[:-1]
        # affine projection list
        # in_dim: [in_features]+out_features[:-1]
        # out_dim: out_features
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size)
             for (in_size, out_size) in zip(in_features, out_features)]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def init_layers(self):
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(
                layer.weight,
                gain=torch.nn.init.calculate_gain('relu')
            )

    def forward(self, inputs):
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs


class BatchNormConv1d(nn.Module):
    '''
    conv1d with BatchNorm, set the activation between conv1d and BatchNorm
    BatchNorm is initialized with TF default values for momentum and eps
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 activation=None):
        super(BatchNormConv1d, self).__init__()
        self.padding = padding
        # https://pytorch.org/docs/stable/nn.html?highlight=constantpad1d#torch.nn.ConstantPad1d
        # torch.nn.ConstantPad1d(padding, value), padding(int/list), value(int): value to be padding
        self.padder = nn.ConstantPad1d(padding, 0)
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.99, eps=1e-3)
        self.activation = activation

    def init_layers(self):
        if type(self.activation) == torch.nn.ReLU:
            w_gain = 'relu'
        elif type(self.activation) == torch.nn.Tanh:
            w_gain = 'tanh'
        elif self.activation is None:
            w_gain = 'linear'
        else:
            raise RuntimeError('Unknown activation function')
        torch.nn.init.xavier_uniform_(
            self.conv1d.weight,
            gain=torch.nn.init.calculate_gain(w_gain)
        )

    def forward(self, x):
        x = self.padder(x)
        x = self.conv1d(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)  # value
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)  # gate
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def init_layers(self):
        torch.nn.init.xavier_uniform_(
            self.H.weight,
            gain=torch.nn.init.calculate_gain('relu')
        )
        torch.nn.init.xavier_uniform_(
            self.T.weight,
            gain=torch.nn.init.calculate_gain('sigmoid')
        )

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1 - T)


class CBHG(nn.Module):
    '''
    CBHG module: a recurrent neural network composed of:
    - conv1d banks
    - highway networks + residual connections
    - bidirectional gated recurrent units

    Args:
        in_features(int): sample_size
        K(int): max filter size in conv bank
        projections(list): conv channel size for conv projections
        num_highways(int): number of highway layers

    Shapes:
        - input: [batch_size,time,dim]
        - output: [batch_size,time,dim*2]
    '''

    def __init__(self,
                 in_features,
                 K=16,
                 conv_bank_features=128,
                 conv_projections=(128, 128),
                 highway_features=128,
                 gru_features=128,
                 num_highways=4):
        super(CBHG, self).__init__()
        self.in_features = in_features
        self.conv_bank_features = conv_bank_features
        self.highway_features = highway_features
        self.gru_features = gru_features
        self.conv_projections = conv_bank_features
        self.relu = nn.ReLU()
        # list of conv1d bank with filter size k=1,2,...,K
        self.conv1d_banks = nn.ModuleList([
            BatchNormConv1d(
                in_features, conv_bank_features,
                kernel_size=k,
                stride=1,
                padding=[(k - 1) // 2, k // 2],
                activation=self.relu
            ) for k in range(1, K + 1)
        ])
        # max pooling of conv bank, with padding
        self.max_pool1d = nn.Sequential(
            nn.ConstantPad1d([0, 1], value=0),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=0)
        )
        out_features = [K * conv_bank_features] + conv_projections[:-1]
        activations = [self.relu] * (len(conv_projections) - 1)
        activations += [None]
        layer_set = []
        for (in_size, out_size, ac) in zip(out_features, conv_projections,
                                           activations):
            layer = BatchNormConv1d(
                in_size,
                out_size,
                kernel_size=3,
                stride=1,
                padding=[1, 1],
                activation=ac
            )
            layer_set.append(layer)
        self.conv1d_projections = nn.ModuleList(layer_set)
        # if output_size of conv1d is not equal to highway_features,
        # then linear projection to highway_features
        if self.highway_features != conv_projections[-1]:
            self.pre_highway = nn.Linear(conv_projections[-1], highway_features, bias=False)
        self.highways = nn.ModuleList([
            Highway(highway_features, highway_features)
            for _ in range(num_highways)
        ])
        # bi-directional GRU layer
        # https://pytorch.org/docs/stable/nn.html?highlight=gru#torch.nn.GRU
        self.gru = nn.GRU(
            input_size=gru_features,
            hidden_size=gru_features,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, inputs):
        # (batch_size,T_in,in_features)
        x = inputs
        # perform conv1d on time-axis,
        # so transpose matrix to (batch_size,in_features,T_in)
        if x.size(-1) == self.in_features:
            x = x.transpose(1, 2)
        outs = []
        for conv1d in self.conv1d_banks:
            out = conv1d(x)
            outs.append(out)
        # concat conv1d bank outputs
        # [batch_size,hid_features*K,T_in]
        x = torch.cat(outs, dim=1)
        assert x.size(1) == self.conv_bank_features * len(self.conv1d_banks)
        x = self.max_pool1d(x)
        for conv1d in self.conv1d_projections:
            x = conv1d(x)
        # [batch_size,T_in,hid_features]
        x = x.transpose(1, 2)
        # back to the original shape
        # res-connection in CBHG,
        # why linear projection to ensure last dim of x&inputs?
        x += inputs
        # ensure last dim of x(output of conv1d layers) to highway_features
        if self.highway_features != self.conv1d_projections[-1]:
            x = self.pre_highway(x)
        for highway in self.highways:
            x = highway(x)
        # for fast computation on GPU
        self.gru.flatten_parameters()
        # (output,h_n)
        # containing the output features h_t from the last layer of the GRU, for each time step
        # output(batch first): [batch_size,seq_len,num_directions*hidden_size]
        # containing the hidden state of all layer for t=seq_len(last time step)
        # h_n(batch_size): [batch_size,num_directions*number_layers,hidden_size]
        outputs, _ = self.gru(x)
        return outputs


class EncoderCBHG(nn.Module):
    def __init__(self):
        super(EncoderCBHG, self).__init__()
        self.cbhg = CBHG(
            in_features=128,
            K=16,
            conv_bank_features=128,
            conv_projections=(128, 128),
            highway_features=128,
            gru_features=128,
            num_highways=4
        )

    def forward(self, inputs):
        return self.cbhg(inputs)


class Encoder(nn.Module):
    def __init__(self, in_features):
        super(Encoder, self).__init__()
        self.prenet = Prenet(in_features=in_features, out_features=(256, 128))
        self.cbhg = EncoderCBHG()

    def forward(self, inputs):
        '''
        inputs: [batch_size,time,in_features(embedding_size)]
        outputs: [batch_size,time,out_features*num_directions(2)]
        :param inputs: should be input embedding
        :return:
        '''
        inputs = self.prenet(inputs)
        return self.cbhg(inputs)


class PostCBHG(nn.Module):
    def __init__(self, mel_dim):
        super(PostCBHG, self).__init__()
        self.cbhg = CBHG(
            in_features=mel_dim,
            K=8,
            conv_bank_features=128,
            conv_projections=[256, mel_dim],
            highway_features=128,
            gru_features=128,
            num_highways=4
        )

    def forward(self, inputs):
        return self.cbhg(inputs)


class Decoder(nn.Module):
    '''
    Decoder module
    Args:
        in_feature(int): input_size of input tensor(encoder output)
        memory_dim(int): memory_size of memory tensor(previous time step output)
        r(int): number of outputs per time step
    '''

    def __init__(self, in_features, memory_dim, r):
        super(Decoder, self).__init__()
        self.r = r
        self.in_features = in_features
        self.max_decoder_steps = 500
        self.memory_dim = memory_dim
        # memory -> [Prenet] -> processed_memory
        self.prenet = Prenet(memory_dim * r, out_features=[256, 128])
        # processded_inputs, processed_memory -> |Attention| -> Attention, attention, RNN state
        self.attention_rnn = AttentionRNNCell(
            out_dim=128,
            rnn_dim=256,
            annot_dim=in_features,
            memory_dim=128,
            align_model='ls'
        )
        self.project_to_decoder_in = nn.Linear(256 + in_features, 256)
        self.decoder_rnns = nn.ModuleList(
            [nn.GRUCell(256, 256) for _ in range(2)]
        )
        self.project_to_mel = nn.Linear(256, memory_dim * r)
        self.stopnet = StopNet(256 + memory_dim * r)

    def init_layers(self):
        torch.nn.init.xavier_uniform_(
            self.project_to_decoder_in.weight,
            gain=torch.nn.init.calculate_gain('linear')
        )
        torch.nn.init.xavier_uniform_(
            self.project_to_mel.weight,
            gain=torch.nn.init.calculate_gain('linear')
        )

    def _reshape_memory(self, memory):
        B = memory.shape[0]
        # group multiple frames if necessary
        if memory.size(-1) == self.memory_dim:
            memory = memory.contiguous()
            # [batch_size,T_decoder*r,-1] -> [batch_size,T_decoder,-1]
            memory = memory.view(B, memory.size(1) // self.r, -1)
        # time first -> [T_decoder,batch_size,memory_dim]
        memory = memory.transpose(0, 1)
        return memory

    def forward(self, inputs, memory=None, mask=None):
        '''
        decoder forward step.
        If decoder inputs are not given(e.g., at testing time), as noted in Tacotron paper,
        **greedy decoding** is adapted.
        :param inputs: encoder outputs, [batch_size,time,encoder_out_dim]
        :param memory: decoder memory(auto-regressive), [batch_size,#mel_specs,mel_spec_dim]
        if None(at eval-time), the last decoder outputs are used as decoder inputs
        :param mask: attention mask for sequence padding
        :return:
        '''
        B = inputs.size(0)
        T = inputs.size(1)

        if memory is not None:
            memory = self._reshape_memory(memory)
            T_decoder = memory.size(0)
        # go frame as zero matrix
        initial_memory = inputs.data.new(B, self.memory_dim * self.r).zero_()
        # decoder states
        attention_rnn_hidden = inputs.data.new(B, 256).zero_()
        decoder_rnn_hidden = [
            inputs.data.new(B, 256).zero_()
            for _ in range(len(self.decoder_rnns))
        ]
        current_context_vec = inputs.data.new(B, self.in_features).zero_()
        attention = inputs.data.new(B, T).zero_()
        attention_cum = inputs.data.new(B, T).zero_()
        outputs = []
        attentions = []
        stop_tokens = []
        t = 0
        memory_input = initial_memory
        while True:
            if t > 0:
                if memory is None:
                    memory_input = outputs[-1]  # the last output
                else:
                    memory_input = memory[t - 1]
                # prenet
                processed_memory = self.prenet(memory_input)
                # attention rnn
                attention_cat = torch.cat(
                    (attention.unsqueeze(1), attention_cum.unsqueeze(1)),
                    dim=1
                )
                attention_rnn_hidden, current_context_vec, attention = self.attention_rnn(
                    processed_memory, current_context_vec, attention_rnn_hidden,
                    inputs, attention_cat, mask, t
                )
                attention_cum += attention
                # concat rnn output and attention context vector
                decoder_input = self.project_to_decoder_in(
                    torch.cat((attention_rnn_hidden, current_context_vec), -1)
                )
                for idx in range(len(self.decoder_rnns)):
                    decoder_rnn_hidden[idx] = self.decoder_rnns[idx](
                        decoder_input, decoder_rnn_hidden[idx]
                    )
                    # residual connection
                    decoder_input = decoder_rnn_hidden[idx] + decoder_input
                decoder_output = decoder_input
                # predict mel vector form decoder output
                output = self.project_to_mel(decoder_output)
                output = torch.sigmoid(output)
                stopnet_input = torch.cat([decoder_input, output], -1)
                stop_token = self.stopnet(stopnet_input)
                outputs += [output]
                attentions += [attention]
                stop_tokens += [stop_token]
                t += 1
                if memory is not None:
                    if t >= T_decoder:
                        break
                else:
                    if t > inputs.shape[1] / 4 and stop_token > 0.6:
                        break
                    elif t > self.max_decoder_steps:
                        print('decoder stopped with max_decoder_steps')
                        break
        # back to batch first
        # [T_decoder,batch_size,dim] -> [batch_size,T_decoder,dim]
        attentions = torch.stack(attentions).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1)
        return outputs, attentions, stop_tokens


class StopNet(nn.Module):
    def __init__(self, in_features):
        super(StopNet, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()
        torch.nn.init.xavier_uniform_(
            self.linear.weight,
            gain=torch.nn.init.calculate_gain('linear')
        )

    def forward(self, inputs):
        x = inputs
        x = self.dropout(x)
        x = self.linear(x)
        outputs = self.sigmoid(x)
        return outputs
