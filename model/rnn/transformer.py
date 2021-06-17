import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def get_transformer_mask(x, y, device):
    ones = torch.ones((x, y)).to(device)
    tril_mask = torch.tril(ones)
    return tril_mask


class Positional_Encoding(nn.Module):
    def __init__(self, dimension, p=0.1):
        super().__init__()
        max_len = 5000
        self.dropout = nn.Dropout(p=p)

        pe = torch.zeros(max_len, dimension)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, dimension, 2).float() * (-math.log(10000.0) / dimension))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # pe.size() -> [1, max_len, dimension]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        input: (batch_size, sequence_length, dimension)
        output: (batch_size, sequence_length, dimension)
        """
        output = x + self.pe[:, :x.size(1), :]
        output = self.dropout(output)
        return output


class Transformer(nn.Module):
    def __init__(self, encoder_input_size, decoder_input_size, hidden_size=512, num_layers=6, **kwargs):
        super().__init__()
        nhead = kwargs.get("nhead", 8)
        p = kwargs.get("transformer_dropout", 0.1)

        self.src_fc = nn.Linear(encoder_input_size, hidden_size)
        self.src_pe = Positional_Encoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=p)
        encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        # self.encoder = nn.Transformer(hidden_size)

        self.tgt_fc = nn.Linear(decoder_input_size, hidden_size)
        self.tgt_pe = Positional_Encoding(hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead, dropout=p)
        decoder_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)
        

    def forward(self, src, tgt=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Inputs:
            src : (batch_size, source_sequence_length, input_size)
            -> TransformerのEncoder側の入力
            tgt : (batch_size, target_sequence_length, input_size)
            -> TransformerのDecoder側の入力
        Outputs:
            output : (batch_size, target_sequence_length, hidden_size)
            -> Transformerの最終出力
        """
        if tgt is None:
            tgt = src
        
        src = self.src_fc(src)
        """ src : (batch_size, source_sequence_length, input_size) """ 
        src_pe = self.src_pe(src).permute(1, 0, 2)
        """ src_pe : (source_sequence_length, batch_size, hidden_size) """ 
        if src_key_padding_mask is not None:
            bs, sl = src_key_padding_mask.size()
            memory = self.encoder(
                src_pe,
                mask=get_transformer_mask(sl, sl, src_pe.device),
                # src_key_padding_mask=src_key_padding_mask.bool()
            )
        else:
            memory = self.encoder(src_pe)
        """ memory : (source_sequence_length, batch_size, hidden_size) """

        tgt = self.tgt_fc(tgt)
        """ tgt : (batch_size, target_sequence_length, input_size) """
        tgt_pe = self.tgt_pe(tgt).permute(1, 0, 2)
        """ tgt_pe : (target_sequence_length, batch_size, hidden_size) """
        if src_key_padding_mask is not None and tgt_key_padding_mask is not None:
            bs, sl = src_key_padding_mask.size()
            bs, tl = tgt_key_padding_mask.size()
            output = self.decoder(
                tgt_pe, memory,
                tgt_mask=get_transformer_mask(tl, tl, tgt_pe.device),
                memory_mask=get_transformer_mask(tl, sl, memory.device),
                # tgt_key_padding_mask=tgt_key_padding_mask.bool(),
                # memory_key_padding_mask=src_key_padding_mask.bool()
            )
        else:
            output = self.decoder(tgt_pe, memory)
        """ output : (target_sequence_length, batch_size, hidden_size) """
        output = output.permute(1, 0, 2)
        """ output : (batch_size, target_sequence_length, hidden_size) """
        memory = memory.permute(1, 0, 2)
        """ memory : (batch_size, source_sequence_length, hidden_size) """
        return output, memory



class Transformer_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=6, **kwargs):
        super().__init__()
        nhead = kwargs.get("nhead", 8)
        p = kwargs.get("transformer_dropout", 0.1)

        self.src_fc = nn.Linear(input_size, hidden_size)
        self.src_pe = Positional_Encoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=p)
        encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        # self.encoder = nn.Transformer(hidden_size)
                

    def forward(self, src, src_key_padding_mask=None):
        """
        Inputs:
            src : (batch_size, source_sequence_length, input_size)
            -> TransformerのEncoder側の入力
        Outputs:
            output : (batch_size, source_sequence_length, hidden_size)
            -> Transformerの最終出力
        """
        
        src = self.src_fc(src)
        """ src : (batch_size, source_sequence_length, input_size) """ 
        src_pe = self.src_pe(src).permute(1, 0, 2)
        """ src_pe : (source_sequence_length, batch_size, hidden_size) """    
        if src_key_padding_mask is not None:
            bs, sl = src_key_padding_mask.size()
            output = self.encoder(
                src_pe,
                mask=get_transformer_mask(sl, sl, src_pe.device),
                # src_key_padding_mask=src_key_padding_mask.bool()
            )
        else:
            output = self.encoder(src_pe)
        """ output : (source_sequence_length, batch_size, hidden_size) """
        output = output.permute(1, 0, 2)
        """ output : (batch_size, source_sequence_length, hidden_size) """
        return output


class Transformer_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=6, **kwargs):
        super().__init__()
        nhead = kwargs.get("nhead", 8)
        p = kwargs.get("transformer_dropout", 0.1)

        self.tgt_fc = nn.Linear(input_size, hidden_size)
        self.tgt_pe = Positional_Encoding(hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead, dropout=p)
        decoder_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)
        

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Inputs:
            src : (batch_size, source_sequence_length, hidden_size)
            -> TransformerのEncoder側の出力
            tgt : (batch_size, target_sequence_length, input_size)
            -> TransformerのDecoder側の入力
        Outputs:
            output : (batch_size, target_sequence_length, hidden_size)
            -> Transformerの最終出力
        """
        
        src = src.permute(1, 0, 2)
        """ src : (source_sequence_length, batch_size, hidden_size) """        

        tgt = self.tgt_fc(tgt)
        """ tgt : (batch_size, target_sequence_length, input_size) """
        tgt_pe = self.tgt_pe(tgt).permute(1, 0, 2)
        """ tgt : (target_sequence_length, batch_size, hidden_size) """
        if src_key_padding_mask is not None and tgt_key_padding_mask is not None:
            bs, sl = src_key_padding_mask.size()
            bs, tl = tgt_key_padding_mask.size()
            output = self.decoder(
                tgt_pe, src,
                tgt_mask=get_transformer_mask(tl, tl, tgt_pe.device),
                memory_mask=get_transformer_mask(tl, sl, memory.device),
                # tgt_key_padding_mask=tgt_key_padding_mask.bool(),
                # memory_key_padding_mask=src_key_padding_mask.bool()
            )
        else:
            output = self.decoder(tgt_pe, src)
        """ output : (target_sequence_length, batch_size, hidden_size) """
        output = output.permute(1, 0, 2)
        """ output : (batch_size, target_sequence_length, hidden_size) """
        return output