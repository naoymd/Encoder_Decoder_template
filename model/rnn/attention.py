import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from typing import Dict, Any, Tuple


def draw_heatmap(data_map, row_labels, column_labels, save_dir=None, name=None):
    fig, ax = plt.subplots(figsize=(20, 4))
    heatmap = ax.pcolor(data_map.T, cmap=plt.cm.Blues)

    ax.set_xticks(np.arange(data_map.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data_map.shape[1]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False, rotation=90)
    ax.set_yticklabels(column_labels, minor=False)
    plt.savefig(os.path.join(save_dir, name + '.png'), bbox_inches='tight')
    plt.close()


class Attention(nn.Module):
    def __init__(self, dimension: int, **kwargs: Dict[str, Any]):
        super(Attention, self).__init__()
        self.qkv_linear = kwargs.get('qkv_linear', True)
        self.query_linear = nn.Linear(dimension, dimension)
        self.key_linear = nn.Linear(dimension, dimension)
        self.value_linear = nn.Linear(dimension, dimension)

        self.mask = kwargs.get('mask', False)
        self.scaled = kwargs.get('scaled', True)
        self.scale = 1.0 / math.sqrt(dimension)
        self.softmax = nn.Softmax(dim=-1)

        self.output_mode = kwargs.get('output_mode', '')
        self.conv = nn.Conv1d(
            dimension*2, dimension,
            kernel_size=1, stride=1, dilation=1, padding=0, groups=2
        )
        self.bn = nn.BatchNorm1d(dimension)
        self.pool = nn.AdaptiveAvgPool1d(dimension)
        self.cat_linear = nn.Linear(dimension*2, dimension)
        self.layernorm = nn.LayerNorm(dimension)
        self.output_linear = nn.Linear(dimension, dimension)
        self.tanh = nn.Tanh()

        init.xavier_uniform_(self.query_linear.weight)
        init.xavier_uniform_(self.key_linear.weight)
        init.xavier_uniform_(self.value_linear.weight)
        init.xavier_uniform_(self.cat_linear.weight)
        init.xavier_uniform_(self.output_linear.weight)

    def subsequent_mask(self, attention_map):
        # To Do
        mask = torch.ones_like(attention_map).tril().bool()
        return mask

    def pad_mask(self, attention_map, query, memory):
        # To Do
        mask = None
        return mask

    def forward(self, input: torch.Tensor, memory: torch.Tensor, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            input : (batch_size, output_size, dimension) or (batch_size, dimension)
            -> memoryの情報をattentionされる特徴量
            memory : (batch_size, queried_size, dimension)
            -> inputにattentionする特徴量
        Outputs:
            output : (batch_size, output_size, dimension)
            -> attentionされたinput特徴量
            attention_map : (batch_size, output_size, queried_size)
            -> attention map
        """
        query = input
        key = memory
        value = memory

        query_squeeze = False
        if len(query.size()) == 2:
            input = input.unsqueeze(dim=1)
            query = query.unsqueeze(dim=1)
            query_squeeze = True

        if self.qkv_linear:
            query = self.query_linear(query)
            key = self.key_linear(key)
            value = self.value_linear(value)
        """
        query : (batch_size, output_size, dimension) or (batch_size, 1, dimension)
        key : (batch_size, queried_size, dimension)
        value: (batch_size, queried_size, dimension)
        """
        """
        attention_map :
            (batch_size, output_size, dimension) * (batch_size, dimension, queried_size)
            -> (batch_size, output_size, queried_size)
        """
        attention_map = torch.matmul(query, key.permute(0, 2, 1))
        if self.scaled:
            attention_map = attention_map.mul_(self.scale)
        attention_map = self.softmax(attention_map)
        if self.mask and input != memory:
            #To Do (mask for <'pad'>)
            # mask = self.pad_mask()
            attention_map = attention_map * mask
            pass
        if self.mask and input == memory:
            # To Do (mask for self attention)
            # mask = self.pad_mask()
            # attention_map = attention_map * mask
            mask = self.subsequent_mask(attention_map)
            attention_map = attention_map * mask
        """
        output :
            (batch_size, output_size, queried_size) * (batch_size, queried_size, dimension) 
            -> (batch_size, output_size, dimension)
        (option)
            cat : output と query のconcat
            add : output と query の和
        """
        output = torch.matmul(attention_map, value)
        if self.output_mode == 'concat':
            output = torch.cat([output, input], dim=2)
            output = self.pool(F.relu(self.bn(self.conv(output.permute(0, 2, 1))).permute(0, 2, 1)))
            # output = F.relu(self.cat_linear(output))
        elif self.output_mode == 'add':
            output = output + input
            # output = self.layernorm(output + input)
            # output = self.output_linear(output)
        else:
            # output = self.output_linear(output)
            pass
        output = self.output_linear(output)
        output = self.tanh(output)
        if query_squeeze:
            output = output.squeeze(dim=1)
        """
        output:
            (batch_size, output_size, dimension) or (batch_size, dimension)
        attention_map:
            (batch_size, output_size, queried_size) or (batch_size, 1, queried_size)
        """
        return output, attention_map


class MultiheadAttention(nn.Module):
    def __init__(self, dimension: int, **kwargs: Dict[str, Any]):
        super(MultiheadAttention, self).__init__()
        num_heads = kwargs.get('num_heads', 4)
        self.qkv_linear = kwargs.get('qkv_linear', True)
        self.query_linear = nn.Linear(dimension, dimension)
        self.key_linear = nn.Linear(dimension, dimension)
        self.value_linear = nn.Linear(dimension, dimension)
        self.multihead_attention = nn.MultiheadAttention(dimension, num_heads)
        self.output_mode = kwargs.get('output_mode', '')
        self.conv = nn.Conv1d(
            dimension*2, dimension,
            kernel_size=1, stride=1, dilation=1, padding=0, groups=2
        )
        self.bn = nn.BatchNorm1d(dimension)
        self.pool = nn.AdaptiveAvgPool1d(dimension)
        self.cat_linear = nn.Linear(dimension*2, dimension)
        self.output_linear = nn.Linear(dimension, dimension)
        
        self.tanh = nn.Tanh()

        init.xavier_uniform_(self.query_linear.weight)
        init.xavier_uniform_(self.key_linear.weight)
        init.xavier_uniform_(self.value_linear.weight)
        init.xavier_uniform_(self.cat_linear.weight)
        init.xavier_uniform_(self.output_linear.weight)

    def forward(self, input: torch.Tensor, memory: torch.Tensor, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            input : (batch_size, output_size, dimension) or (batch_size, dimension)
            -> memoryの情報をattentionされる特徴量
            memory : (batch_size, queried_size, dimension)
            -> inputにattentionする特徴量
        Outputs:
            output : (batch_size, output_size, dimension)
            -> attentionされたinput特徴量
            attention_map : (batch_size, output_size, queried_size)
            -> attention map
        """
        query = input
        key = memory
        value = memory

        query_squeeze = False
        if len(query.size()) == 2:
            input = input.unsqueeze(dim=1)
            query = query.unsqueeze(dim=1)
            query_squeeze = True

        if self.qkv_linear:
            query = self.query_linear(query)
            key = self.key_linear(key)
            value = self.value_linear(value)

        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        """
        query : (output_size, batch_size, dimension) or (1, batch_size, dimension)
        key : (queried_size, batch_size, dimension)
        value: (queried_size, batch_size, dimension)
        """

        output, attention_map = self.multihead_attention(query, key, value)
        output = output.permute(1, 0, 2)
        """
        output : (batch_size, output_size, dimension)
        attention_map : (batch_size, output_size, queried_size)
        (option)
            cat : output と query のconcat
            add : output と query の和
        """
        if self.output_mode == 'concat':
            output = torch.cat([output, input], dim=2)
            output = self.pool(F.relu(self.bn(self.conv(output.permute(0, 2, 1))).permute(0, 2, 1)))
            # output = F.relu(self.cat_linear(output))
        elif self.output_mode == 'add':
            output = output + input
            # output = self.layernorm(output + input)
            # output = self.output_linear(output)
        else:
            # output = self.output_linear(output)
            pass
        # output = self.output_linear(output)
        # output = self.tanh(output)
        if query_squeeze:
            output = output.squeeze(dim=1)
        """
        output:
            (batch_size, output_size, dimension) or (batch_size, dimension)
        attention_map:
            (batch_size, output_size, queried_size) or (batch_size, 1, queried_size)
        """
        return output, attention_map