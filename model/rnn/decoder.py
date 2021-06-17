import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


"""
Inputs:
    Bidirectional == False:
        x : (batch_size, sequence_length_size, input_size)
        encoder_out : (batch_size, sequence_length_size, hidden_size)
        encoder_h : (batch_size, hidden_size)
    Bidirectional == True:
        x : (batch_size, sequence_length_size, input_size)
        encoder_out : (batch_size, sequence_length_size, hidden_size*2)
        encoder_h : (batch_size, hidden_size*2)
Outputs:
    Bidirectional == False:
        out, (c_out) : (batch_size, sequence_length_size, hidden_size)
        h, (c) : (batch_size, hidden_size)
    Bidirectional == True:
        out, (c_out) : (batch_size, sequence_length_size, hidden_size*2)
        h, (c) : (batch_size, hidden_size*2)
"""

class GRU_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.ModuleList()
        for _ in range(self.num_layers):
            self.gru.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size
        # print(self.gru)

    def forward(self, x, encoder_h, encoder_out=None, seq_length=None):
        seq_len = x.size(1)
        h_n = [encoder_h[i, :, :] for i in range(self.num_layers)]
        out = []
        for i in range(seq_len):
            x_i = x[:, i, :]
            for j, layer in enumerate(self.gru):
                x_i = layer(x_i, h_n[j])
                h_n[j] = x_i
            out.append(x_i)
        out = torch.stack(out, dim=1)
        h = torch.stack(h_n, dim=0)
        return out, h


class Bidirectional_GRU_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_f = nn.ModuleList()
        self.gru_b = nn.ModuleList()
        for _ in range(self.num_layers):
            self.gru_f.append(nn.GRUCell(input_size, hidden_size))
            self.gru_b.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size
        # print(self.gru_f)
        # print(self.gru_b)

    def reverse(self, x):
        reverse_x = torch.flip(x, [1])
        return reverse_x

    def forward(self, x, encoder_h, encoder_out=None, seq_length=None):
        seq_len = x.size(1)
        h_f_n = [encoder_h[i, :, :self.hidden_size] for i in range(self.num_layers)]
        h_b_n = [encoder_h[i, :, self.hidden_size:] for i in range(self.num_layers)]
        reverse_x = self.reverse(x)
        out_f = []
        out_b = []
        for i in range(seq_len):
            x_f_i = x[:, i, :]
            x_b_i = reverse_x[:, i, :]
            for j, (layer_f, layer_b) in enumerate(zip(self.gru_f, self.gru_b)):
                x_f_i = layer_f(x_f_i, h_f_n[j])
                x_b_i = layer_b(x_b_i, h_b_n[j])
                h_f_n[j] = x_f_i
                h_b_n[j] = x_b_i
            out_f.append(x_f_i)
            out_b.append(x_b_i)
        out_f = torch.stack(out_f, dim=1)
        out_b = torch.stack(out_b, dim=1)
        out_b = self.reverse(out_b)
        out = torch.cat([out_f, out_b], dim=2)
        h_f = torch.stack(h_f_n, dim=0)
        h_b = torch.stack(h_b_n, dim=0)
        h = torch.cat([h_f, h_b], dim=2)
        return out, h


class GRU_Decoder_(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, bidirectional=False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.padidx = 0
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers = num_layers,
            bidirectional = bidirectional,
            batch_first = True
        )

    def forward(self, x, encoder_h, encoder_out=None, seq_length=None):
        batch_size = x.size(0)
        encoder_h = encoder_h.reshape(-1, batch_size, self.hidden_size)
        if seq_length is not None:
            # x: PackedSequence of (batch_size, seq_length, input_size)
            x = pack_padded_sequence(x, seq_length, batch_first=True, enforce_sorted=False)
        output, h = self.gru(x, encoder_h)
        h = h.reshape(self.num_layers, batch_size, -1)
        # output: (batch_size, seq_length, hidden_size)
        # h: (num_layers, batch_size, hidden_size)
        if seq_length is not None:
            output, lengths = pad_packed_sequence(output, batch_first=True, padding_value=self.padidx)
        return output, h


class LSTM_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.ModuleList()
        for _ in range(self.num_layers):
            self.lstm.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        # print(self.lstm)

    def forward(self, x, encoder_h, encoder_out=None, seq_length=None):
        seq_len = x.size(1)
        encoder_h, encoder_c = encoder_h
        h_n = [encoder_h[i, :, :] for i in range(self.num_layers)]
        c_n = [encoder_c[i, :, :] for i in range(self.num_layers)]
        out = []
        for i in range(seq_len):
            x_i = x[:, i, :]
            for j, layer in enumerate(self.lstm):
                x_i, c_i = layer(x_i, (h_n[j], c_n[j]))
                h_n[j] = x_i
                c_n[j] = c_i
            out.append(x_i)
        out = torch.stack(out, dim=1)
        h = torch.stack(h_n, dim=0)
        c = torch.stack(c_n, dim=0)
        return out, (h, c)


class Bidirectional_LSTM_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_f = nn.ModuleList()
        self.lstm_b = nn.ModuleList()
        for _ in range(self.num_layers):
            self.lstm_f.append(nn.LSTMCell(input_size, hidden_size))
            self.lstm_b.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        # print(self.lstm_f)
        # print(self.lstm_b)

    def reverse(self, x):
        reverse_x = torch.flip(x, [1])
        return reverse_x

    def forward(self, x, encoder_h, encoder_out=None, seq_length=None):
        seq_len = x.size(1)
        encoder_h, encoder_c = encoder_h
        h_f_n = [encoder_h[i, :, :self.hidden_size] for i in range(self.num_layers)]
        h_b_n = [encoder_h[i, :, self.hidden_size:] for i in range(self.num_layers)]
        c_f_n = [encoder_c[i, :, :self.hidden_size] for i in range(self.num_layers)]
        c_b_n = [encoder_c[i, :, self.hidden_size:] for i in range(self.num_layers)]
        reverse_x = self.reverse(x)
        out_f = []
        out_b = []
        for i in range(seq_len):
            x_f_i = x[:, i, :]
            x_b_i = reverse_x[:, i, :]
            for j, (layer_f, layer_b) in enumerate(zip(self.lstm_f, self.lstm_b)):
                x_f_i, c_f_i = layer_f(x_f_i, (h_f_n[j], c_f_n[j]))
                x_b_i, c_b_i = layer_b(x_b_i, (h_b_n[j], c_b_n[j]))
                h_f_n[j] = x_f_i
                h_b_n[j] = x_b_i
                c_f_n[j] = c_f_i
                c_b_n[j] = c_b_i
            out_f.append(x_f_i)
            out_b.append(x_b_i)
        out_f = torch.stack(out_f, dim=1)
        out_b = torch.stack(out_b, dim=1)
        out_b = self.reverse(out_b)
        out = torch.cat([out_f, out_b], dim=2)
        h_f = torch.stack(h_f_n, dim=0)
        h_b = torch.stack(h_b_n, dim=0)
        c_f = torch.stack(c_f_n, dim=0)
        c_b = torch.stack(c_b_n, dim=0)
        h = torch.cat([h_f, h_b], dim=2)
        c = torch.cat([c_f, c_b], dim=2)
        return out, (h, c)


class LSTM_Decoder_(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.padidx = 0
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers = num_layers,
            bidirectional = bidirectional,
            batch_first = True
        )
        if bidirectional:
            hidden_size = hidden_size * 2
        else:
            hidden_size = hidden_size

    def forward(self, x, encoder_h, encoder_out=None, seq_length=None):
        batch_size = x.size(0)
        encoder_h, encoder_c = encoder_h
        encoder_h = encoder_h.reshape(-1, batch_size, self.hidden_size)
        encoder_c = encoder_c.reshape(-1, batch_size, self.hidden_size)
        if seq_length is not None:
            # x: PackedSequence of (batch_size, seq_length, input_size)
            x = pack_padded_sequence(x, seq_length, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.lstm(x, (encoder_h, encoder_c))
        h = h.reshape(self.num_layers, batch_size, -1)
        c = c.reshape(self.num_layers, batch_size, -1)
        # output: (batch_size, seq_length, hidden_size)
        # h, c: (num_layers, batch_size, hidden_size)
        if seq_length is not None:
            output, lengths = pad_packed_sequence(output, batch_first=True, padding_value=self.padidx)
        return output, (h, c)