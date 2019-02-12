# -*- coding: utf-8 -*-
# @Time    : 2019/2/2 14:09
# @Author  : MengnanChen
# @FileName: attention.py
# @Software: PyCharm

import torch
from torch import nn


class BahdanauAttention(nn.Module):
    def __init__(self, annot_dim, query_dim, attn_dim):
        super(BahdanauAttention, self).__init__()
        self.query_layer = nn.Linear(in_features=query_dim, out_features=attn_dim, bias=True)
        self.annot_layer = nn.Linear(annot_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, annots, query):
        '''
        :param annots: [batch_size,max_time,dim]
        :param query: [batch_size,1,dim] or [batch_size,dim]
        :return:
        '''
        if query.dim() == 2:
            # ensure query shape to be [batch_size,1,dim]
            query = query.unsqueeze(1)
        # [batch_size,1,hid_dim]
        processed_query = self.query_layer(query)
        processed_annots = self.annot_layer(annots)
        # [batch_size,max_time,1]
        alignment = self.v(torch.tanh(processed_query + processed_annots))
        # [batch_size,max_time]
        return alignment.squeeze(-1)


class LocationSensitiveAttention(nn.Module):
    def __init__(self,
                 annot_dim,
                 query_dim,
                 attn_dim,
                 kernel_size=31,
                 filters=32):
        super(LocationSensitiveAttention, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        padding = [(kernel_size - 1) // 2, (kernel_size - 1) // 2]
        self.loc_conv = nn.Sequential(
            nn.ConstantPad1d(padding, 0),
            nn.Conv1d(
                in_channels=2,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False
            )
        )
        self.loc_linear = nn.Linear(in_features=filters, out_features=attn_dim, bias=True)
        self.query_layer = nn.Linear(query_dim, attn_dim, bias=True)
        self.annot_layer = nn.Linear(annot_dim, attn_dim, bias=True)
        self.v = nn.Linear(annot_dim, 1, bias=False)
        self.processed_annots = None

    def init_layers(self):
        torch.nn.init.xavier_uniform_(
            self.loc_linear.weight,
            gain=torch.nn.init.calculate_gain('tanh')
        )
        torch.nn.init.xavier_uniform_(
            self.query_layer.weight,
            gain=torch.nn.init.calculate_gain('tanh')
        )
        torch.nn.init.xavier_uniform_(
            self.annot_layer.weight,
            gain=torch.nn.init.calculate_gain('tanh')
        )
        torch.nn.init.xavier_uniform_(
            self.v.weight,
            gain=torch.nn.init.calculate_gain('linear')
        )

    def reset(self):
        self.processed_annots = None

    def forward(self, annot, query, loc):
        '''
        in rayhan edition, *location features: [batch_size,max_time,attention_dim]*
        key: [batch_size,max_time,attention_dim]
        query: [batch_size,1,attention_dim]

        :param annot: [batch_size,max_time,dim]
        :param query: [batch_size,1,dim] or [batch_size,dim]
        :param loc: [batch_size,filters,max_time]
        '''
        if query.dim() == 2:
            # ensure query shape to be [batch_size,1,dim]
            query = query.unsqueeze(1)
        # [batch_size,filters,max_time]
        loc_conv = self.loc_conv(loc)
        # [batch_size,max_time,filters]
        loc_conv = loc_conv.transpose(1, 2)
        # [batch_size,max_time,attn_dim]
        processed_loc = self.loc_linear(loc_conv)
        # [batch_size,1,attn_dim]
        processed_query = self.query_layer(query)
        # cache annots
        if self.processed_annots is None:
            # [batch_size,max_time,attn_dim]
            self.processed_annots = self.annot_layer(annot)
        # [batch_size,max_time,1]
        alignments = self.v(
            torch.tanh(processed_query + self.processed_annots + processed_loc)
        )
        # [batch_size,max_time]
        return alignments.squeeze(-1)


class AttentionRNNCell(nn.Module):
    def __init__(self, out_dim, rnn_dim, annot_dim, memory_dim, align_model):
        '''
        attention rnn wrapper
        :param out_dim(int): context vector feature dim
        :param rnn_dim(int): query tensor dim(rnn hidden state dim)
        :param annot_dim(int): key tensor dim(annotation vector feature dim)
        :param memory_dim(int): memory vector(decoder output) feature dim
        :param align_model(str): 'b' for Bahdanau, 'ls' for Location Sensitive Attention
        '''
        super(AttentionRNNCell, self).__init__()
        self.align_model = align_model
        # https://pytorch.org/docs/stable/nn.html?highlight=grucell#torch.nn.GRUCell
        self.rnn_cell = nn.GRUCell(input_size=annot_dim + memory_dim, hidden_size=rnn_dim)
        if align_model == 'b':
            self.alignment_model = BahdanauAttention(annot_dim, rnn_dim, out_dim)
        elif align_model == 'ls':
            self.alignment_model = LocationSensitiveAttention(annot_dim, rnn_dim, out_dim)
        else:
            raise RuntimeError('wrong alignment model: {}, only support "b"(Bahdanau) '
                               'or "ls"(Location Sensitive)'.format(align_model))

    def forward(self, memory, context, rnn_state, annots, atten, mask, t):
        '''
        :param memory: query, [batch_size,1,dim] or [batch_size,dim]
        :param context: previous context, [batch_size,dim]
        :param rnn_state: [batch_size,out_dim]
        :param annots: key, [batch_size,max_time,annot_dim]
        :param atten: [batch_size,2,max_time]
        :param mask: [batch_size,]
        :param t:
        :return:
        '''
        if t == 0:
            self.alignment_model.reset()
        # concat input query and previous context
        # memory: [batch_size,dim]
        # context: [batch_size,dim]
        # rnn_input: [batch_size,dim*2]
        rnn_input = torch.cat((memory, context), dim=-1)
        # feed it to rnn
        # s_i=f(y_{i-1},c_{i},s_{i-1})
        # input: [batch_size,input_size], tensor containing input feature
        # hx: [batch_size,hidden_size], initial hidden state for each element in the batch
        # rnn_output: [batch_size,hidden_size], the next hidden state for each element in the batch
        rnn_output = self.rnn_cell(input=rnn_input, hx=rnn_state)
        # alignment
        # e_{ij}=a(s_{i-1},h_j)
        # [batch_size,max_time]
        if self.align_model == 'b':
            alignment = self.alignment_model(annots, rnn_output)
        else:
            alignment = self.alignment_model(annots, rnn_output, atten)

        if mask is not None:
            mask = mask.view(memory.size(0), -1)
            alignment.masked_fill_(1 - mask, -float('inf'))
        # normalize context weight
        alignment = torch.sigmoid(alignment) / torch.sigmoid(alignment).sum(dim=1).unsqueeze(1)
        # c_i=\sum_{j=1}^{T_x} \alpha_{ij} h_j
        # [batch_size,1,dim]
        context = torch.bmm(alignment.unsqueeze(1), annots)
        context = context.squeeze(1)
        return rnn_output, context, alignment
