import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from model.rnn.encoder import *
from model.rnn.decoder import *
from model.rnn.transformer import *


class Net(nn.Module):
    def __init__(self, encoder_input_size, decoder_input_size, **kwargs):
        super().__init__()
        self.rnn = kwargs.get("rnn", "GRU")
        bidirection = kwargs.get("bidirection", False)
        hidden_size = kwargs.get("hidden_size", 512)
        num_layers = kwargs.get("num_layers", 1)
        output_size = kwargs.get("output_size", 2)
        print("-"*50)
        print("rnn:", self.rnn)
        print("bidirection:", bidirection)
        print("-"*50)

        # self.embed = nn.Embedding.from_pretrained(embeddings=word_embeddings, freeze=True)
        self.encoder_embed = nn.Linear(1, encoder_input_size)
        self.decoder_embed = nn.Linear(1, decoder_input_size)
        # RNN
        if self.rnn == "GRU":
            if bidirection:
                self.encoder = Bidirectional_GRU_Encoder(encoder_input_size, hidden_size, num_layers)
                self.decoder = Bidirectional_GRU_Decoder(decoder_input_size, hidden_size, num_layers)
            else:
                self.encoder = GRU_Encoder(encoder_input_size, hidden_size, num_layers)
                self.decoder = GRU_Decoder(decoder_input_size, hidden_size, num_layers)
        elif self.rnn == "GRU_":
            self.encoder = GRU_Encoder_(encoder_input_size, hidden_size, num_layers, bidirection)
            self.decoder = GRU_Decoder_(decoder_input_size, hidden_size, num_layers, bidirection)
        if self.rnn == "LSTM":
            if bidirection:
                self.encoder = Bidirectional_LSTM_Encoder(encoder_input_size, hidden_size, num_layers)
                self.decoder = Bidirectional_LSTM_Decoder(decoder_input_size, hidden_size, num_layers)
            else:
                self.encoder = LSTM_Encoder(encoder_input_size, hidden_size, num_layers)
                self.decoder = LSTM_Decoder(decoder_input_size, hidden_size, num_layers)
        elif self.rnn == "LSTM_":
            self.encoder = LSTM_Encoder_(encoder_input_size, hidden_size, num_layers, bidirection)
            self.decoder = LSTM_Decoder_(decoder_input_size, hidden_size, num_layers, bidirection)
        elif self.rnn == "Transformer":
            self.transformer = Transformer(encoder_input_size, decoder_input_size, **kwargs)

        if "Transformer" not in self.rnn and bidirection:
            hidden_size = hidden_size * 2
        
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=kwargs.get("dropout", 0.1))
        init.xavier_uniform_(self.linear.weight)


    def get_mask(self, data, data_len):
        """ 
        data: (batch_size, seq_len, *)
        data_len: (batch_size, *)
        """
        batch_size = data.size(0)
        max_length = data.size(1)
        ones = data_len.new_ones(batch_size, max_length)
        range_tensor = ones.cumsum(dim=1)
        mask = (data_len.reshape(batch_size, 1) >= range_tensor).float()
        return mask

    
    def l2norm(self, x, d):
        norm = torch.norm(x, dim=d, keepdim=True)
        return x / norm


    def forward(self, e_in, d_in, e_len=None, d_len=None):
        """
        Input:
            e_in: (batch_size, max_encoder_length)
            d_in: (batch_size, max_decoder_length)
            e_len: (batch_size)
            d_len: (batch_size)
        Output:
            output: (batch_size, max_decoder_length, output_size)
        """
        e_embed = self.encoder_embed(e_in.unsqueeze(dim=-1))
        d_embed = self.dncoder_embed(d_in.unsqueeze(dim=-1))
        if e_len is not None:
            e_mask = self.get_mask(e_embed, e_len)
        if d_len is not None:
            d_mask = self.get_mask(d_embed, d_len)

        if "Transformer" not in self.rnn:
            e_out, e_h = self.encoder(e_embed)
            e_out = self.dropout(e_out)
            d_out, d_h = self.decoder(d_embed, e_h, e_out)
        else:
            d_out, e_out = self.transformer(e_embed, d_embed, e_mask, d_mask)
            # d_out, e_out = self.transformer(e_embed, d_dec_in)
        
        d_out = self.dropout(d_out)
        if d_len is not None:
            output = F.softmax(torch.mul(self.linear(d_out), d_mask.unsqueeze(dim=-1)), dim=1)
        else:
            output = F.softmax(self.linear(d_out), dim=1)
        
        return output
        