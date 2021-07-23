# -*- coding: utf-8 -*-
# Copyright 2019 Christopher Kümmel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers=4, dropout=0.2)

    def forward(self, x, hidden, total_length, input_lengths):
        self.gru.flatten_parameters()

        # size of input_length (1, batch_size) | batch_size / n for dataparallelism
        packed_input = nn.utils.rnn.pack_padded_sequence(x, input_lengths[0], enforce_sorted=False)
        packed_output, hidden = self.gru(packed_input, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, total_length=total_length)

        return output, hidden

    def initHidden(self, batch_size, device):
        return torch.zeros(4, batch_size, self.hidden_size, device=device)  # 4 layer


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_projection = nn.Linear(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=4, dropout=0.2)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output = self.linear_projection(x)
        self.gru.flatten_parameters()  # flatten parameters for multi gpu
        output, hidden = self.gru(output, hidden)
        output = self.output(output)
        return output, hidden

    def initHidden(self, batch_size, device):
        return torch.zeros(4, batch_size, self.hidden_size, device=device)  # 4 layer gru


class AttnDecoderRNN(nn.Module):
    """Attention Decoder RNN"""
    def __init__(self, hidden_size, output_size) -> None:
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_projection = nn.Linear(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.weight = nn.Parameter(torch.rand(hidden_size), requires_grad=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers=4, dropout=0.2)
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)  # 2 -> seq_len

    def forward(self, x, hidden, encoder_outputs, pad_mask):
        """Attention decoder forward step.
        
        Args:
            x (torch.tensor): Input word embedding [seq_len=1, batch_size, embedding_space].
            hidden (torch.tensor): Previous decoder hidden state [num_layers, batch_size, hidden_size]
            encoder_outputs (torch.tensor): Encoder output [max_seq_len, batch_size, hidden_size]
            pad_mask (torch.tensor): Attention padding mask [batch_size, max_seq_len]
        """

        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        # take last hidden layer and repeat it  [batch_size, hidden_size] -> [max_seq_len, batch_size, hidden_size]
        _hidden = hidden[-1].repeat((seq_len, 1, 1))

        # bahdanau concat attention
        energy = torch.tanh(self.attn(torch.cat((_hidden, encoder_outputs), dim=2)))

        # mulitply energy with weigth matrix   [batch_size, 1, hidden_size] x [batch_size, hidden_size, max_seq_len] -> [batch_size, 1, max_seq_len]
        _weight = self.weight.repeat((batch_size, 1)).unsqueeze(1)
        attn_weights = torch.bmm(_weight, energy.permute(1, 2, 0))

        # attention masking | set padded values to neg infinity -> after softmax they will become 0
        attn_weights[~pad_mask.unsqueeze(1)] = float('-inf')

        attn_weights = self.softmax(attn_weights)

        # multiply attention weights with encoder outputs
        # [batch_size, 1, max_seq_len] x [batch_size, max_seq_len, hidden_size] -> [batch_size, 1, hidden_size]
        context = torch.bmm(attn_weights, encoder_outputs.permute(1, 0, 2)).permute(1, 0, 2)

        output = self.linear_projection(x)  # x (seq_len, batch_size, one_hot_encoding_space) -> (seq_len, batch_size, hidden_size)

        output = torch.cat((output, context), dim=2)  # concat hidden states (dim = 2)
        self.gru.flatten_parameters()  # flatten parameters for multi gpu
        output, hidden = self.gru(output, hidden)
        output = self.output(output)
        return output, hidden
