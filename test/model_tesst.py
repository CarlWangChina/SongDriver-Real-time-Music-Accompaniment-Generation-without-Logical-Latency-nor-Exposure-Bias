import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import pickle
from chordPreprocess import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weighted_features = [19, 22, 26]      # 权重特征
structural_chord = [19, 22, 26]       # 结构和弦
is_weighted = [0]            # 是否为权重音 # 0:不是 1:是
is_cadence = 0                        # 是否为终止和弦

basetone2idx = {
    'C':0, 'D':2, 'E':4, 'F':5, 'G':7, 'A':9, 'B':11
}
keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
full_keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

note2token = {
    "Cbb": "Bb",
    "Cb": "B",
    "C": "C",
    "C#": "Db",
    "C##": "D",
    "Dbb": "C",
    "Db": "Db",
    "D": "D",
    "D#": "Eb",
    "D##": "E",
    "Ebb": "D",
    "Eb": "Eb",
    "E": "E",
    "E#": "F",
    "E##": "F#",
    "Fbb": "Eb",
    "Fb": "E",
    "F": "F",
    "F#": "F#",
    "F##": "G",
    "Gbb":"F",
    "Gb": "F#",
    "G": "G",
    "G#": "Ab",
    "G##": "A",
    "Abb": "G",
    "Ab": "Ab",
    "A": "A",
    "A#": "Bb",
    "A##": "B",
    "Bbb": "A",
    "Bb": "Bb",
    "B": "B",
    "B#": "C",
    "B##": "Db",
}

key2chord = pickle.load(open('./key2chord.pkl', 'rb'))

def normalise_key (keyname):
    key, quality = keyname.split('.')
    key = key[0].upper() + key[1:]
    key = note2token[key]
    k, q = key, quality
    q = 'Major' if quality.strip().upper() == 'MAJOR' else 'minor'
    if len(key) == 2:
        if quality.strip().upper() == 'MAJOR': #b
            if key[-1] == '#':
                k = keys[(keys.index(k[0].upper())+1)%7].upper() + 'b'
                # k = k[0].upper()+k[1] if quality.strip().upper() == 'MAJOR' else k[0].lower()+k[1]
            else:
                k = k[0].upper() + k[1]
        else: ##
            if key[-1] == 'b':
                k = keys[(keys.index(k[0].upper())-1+7)%7].lower() + '#'
            else:
                k = k[0].lower() + k[1]

    else:
        k = k.upper() if quality.strip().upper() == 'MAJOR' else k.lower()
    return k+'_'+q


def tone2idx (tonename):
    if len(tonename.strip()) > 1:
        root, shift = tonename[0], tonename[1]
    else:
        root, shift = tonename[0], ''
    index = 0
    baseC = basetone2idx['C']
    index = basetone2idx[root.upper()] - baseC
    if shift == '':
        return index
    elif shift == 'b':
        return (index - 1 + 12) % 12
    elif shift == '#':
        return (index + 1) % 12

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # [batch_size, 1, len_k], False is masked
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # print(pad_attn_mask)
    # print(pad_attn_mask.expand(batch_size, len_q, len_k))
    # [batch_size, len_q, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # Upper triangular matrix
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    subsequence_mask = subsequence_mask.to(device)
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, args):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = args.d_k

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # Fills elements of self tensor with value where mask is True.
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False).to(device)
        self.W_K = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False).to(device)
        self.W_V = nn.Linear(args.d_model, args.d_v * args.n_heads, bias=False).to(device)
        self.fc = nn.Linear(args.n_heads * args.d_v, args.d_model, bias=False).to(device)
        self.scaledDotProductAttention = ScaledDotProductAttention(args).to(device)
        self.n_heads = args.n_heads
        self.d_k = args.d_k
        self.d_v = args.d_v
        self.d_model = args.d_model
        self.layernorm = nn.LayerNorm(self.d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                                     2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.scaledDotProductAttention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        # print(output.is_cuda, residual.is_cuda, attn.is_cuda)
        # x = nn.LayerNorm(self.d_model)(output + residual).to_device()
        return self.layernorm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(args.d_model, args.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(args.d_ff, args.d_model, bias=False)
        )
        self.d_model = args.d_model
        self.layernorm = nn.LayerNorm(self.d_model)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        residual = residual.to(device)
        output = self.fc(inputs)
        output = output.to(device)
        # [batch_size, seq_len, d_model]
        return self.layernorm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # print(enc_inputs)
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_self_attn_mask = enc_self_attn_mask.to(device)
        enc_inputs = enc_inputs.to(device)
        self.enc_self_attn = self.enc_self_attn.to(device)
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        # enc_outputs: [batch_size, src_len, d_model]
        attn = attn.to(device)
        enc_outputs = self.pos_ffn(enc_outputs)
        enc_outputs = enc_outputs.to(device)
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(args)
        self.dec_enc_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(
            dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.d_model = args.d_model
        self.src_emb = nn.Embedding(args.src_vocab_size, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model, args.dropout)
        self.layers = nn.ModuleList([EncoderLayer(args)
                                    for _ in range(args.n_layers)])

    def forward(self, enc_inputs, keys):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        global structural_chord
        global weighted_features
        global is_weighted
        global is_cadence
        enc_i_size = torch.Size([enc_inputs.size()[0], enc_inputs.size()[1]])
        enc_melody = []
        keyname = str(full_keys[int(keys[0][0])]) + ('.Major' if int(keys[0][1]) == 1 else '.minor')
        for idx1, bar in enumerate(enc_inputs):
            line = []
            for idx2, item in enumerate(bar):
                line.append(id_to_melody[int(item)])
            # print(len(line))
            enc_melody.append(line.copy())
        enc_inputs = enc_inputs.to(device)
        enc_outputs = self.src_emb(enc_inputs)
        dst_size = torch.Size(
            [enc_outputs.size()[0], enc_outputs.size()[1], enc_outputs.size()[2]])
        enc_outputs_new = torch.zeros(dst_size)
        enc_outputs_new = enc_outputs_new.to(device)
        for idx1, bar in enumerate(enc_outputs):
            try:
                # print(keys[0], keys[1])
                isMajor = True if int(keys[idx1][1]) == 1 else False
                basetone = int(keys[idx1][0])
                pushMelody(enc_melody[idx1])
            except Exception as e:
                print(f"Unable to push_melody error: {e}")
                pass
            # print(f'structural_chord:{weighted_features}')
            # weighted_features_all = ''
            # for i in weighted_features:
                # weighted_features_all += str(i)
            # print("all:",weighted_features, weighted_features_all)
            # weighted_features_token = [chord_to_id[weighted_features_all]]
            emb_weighted_features = self.src_emb(torch.tensor(
                weighted_features).to(device))  # [feature_size, d_model-1]
            # print(f"{idx1}: {weighted_features}")
            for idx2, emb_vec in enumerate(bar):
                for feat in emb_weighted_features:
                    # 音符与权重特征拼接 [src_len+1, d_model-1]
                    proj = nn.Linear(len(feat), 25).to(device)
                    feat = feat.to(device)
                    feat = proj(feat)
                    emb_vec = torch.cat((emb_vec, feat))
                projection = nn.Linear(len(emb_vec), self.d_model-1)
                projection = projection.to(device)
                emb_vec = projection(emb_vec)  # 投影到d_model-1维度
                comb_vec = torch.zeros(torch.Size([256]), dtype=torch.float32)
                comb_vec[:self.d_model-1] = emb_vec
                comb_vec[self.d_model-1] = is_weighted[0] if len(is_weighted)>0 else 0
                enc_outputs_new[idx1][idx2] = comb_vec
        enc_outputs = self.pos_emb(enc_outputs_new.transpose(
            0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(
            enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        # print(enc_self_attn_mask)
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            # print(f'355::{enc_outputs.is_cuda}, {enc_self_attn_mask.is_cuda}')
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attn.to(device)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.d_model = args.d_model
        self.tgt_emb = nn.Embedding(args.tgt_vocab_size, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model)
        self.layers = nn.ModuleList([DecoderLayer(args)
                                    for _ in range(args.n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs, keys):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        # [batch_size, tgt_len, d_model-1]
        global structural_chord
        global weighted_features
        global is_weighted
        global is_cadence
        # 先处理结构和弦
        dec_inputs = dec_inputs.to(device)
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs.to(device)
        dst_size = torch.Size([dec_outputs.size()[0], dec_outputs.size()[
                              1], dec_outputs.size()[2]])
        dec_outputs_new = torch.zeros(dst_size)
        # dec_outputs_new = dec_outputs.to(device)
        enc_chord = []
        # print("before", structural_chord)
        for idx1, bar in enumerate(dec_inputs):
            # print(f"bar:{bar}")
            line = []
            # print(f'dec:inputs: {dec_inputs[idx1]}')
            for idx2, item in enumerate(bar):
                line.append(id_to_chord[int(item)])
            enc_chord.append(line)
        for idx1, bar in enumerate(dec_outputs):
            if enc_chord[idx1][1] == 'P' or enc_chord[idx1][1] == 'E':
                # print("condition1")
                keyname = full_keys[int(keys[idx1][0])] + ('.Major' if int(keys[idx1][1]) == 1 else '.minor')
                std_chord = key2chord[normalise_key(keyname)]
                chord_str = ''
                for note in std_chord:
                    chord_str += str(note)
                # pushChord(chord_str)
                pushChord(chord_str)
                structural_chord = [chord_to_id[chord_str]]
            else:
                # print(f'enc_chord:{enc_chord[idx1][1]}')
                pushChord(enc_chord[idx1][1])
            # print(f'structural_chord:{structural_chord}')
            structural_chord_all = ''
            # print(structural_chord)
            # for i in structural_chord:
                # structural_chord_all += str(i)
            # structural_chord_token = [chord_to_id[structural_chord_all]]
            emb_structural_chord = self.tgt_emb(torch.tensor(structural_chord).to(device))
            for idx2, emb_vec in enumerate(bar):
                for feat in emb_structural_chord:
                    # 音符与权重特征拼接 [src_len+1, d_model-1]
                    proj = nn.Linear(len(feat), 25).to(device)
                    feat = feat.to(device)
                    feat = proj(feat)
                    emb_vec = torch.cat((emb_vec, feat))
                # print("size:", emb_vec.size())
                projection = nn.Linear(len(emb_vec), self.d_model - 1)
                projection = projection.to(device)
                emb_vec = projection(emb_vec)
                comb_vec = torch.zeros(torch.Size([256]), dtype=torch.float32)
                comb_vec[:self.d_model - 1] = emb_vec
                comb_vec[self.d_model - 1] = is_cadence
                dec_outputs_new[idx1][idx2] = comb_vec

        dec_outputs_new = dec_outputs_new.to(device)
        dec_outputs = self.pos_emb(dec_outputs_new.transpose(
            0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(
            dec_inputs, dec_inputs)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(
            dec_inputs)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_pad_mask.to(device)
        dec_self_attn_subsequence_mask.to(device)
        # print(dec_self_attn_pad_mask.is_cuda, dec_self_attn_subsequence_mask.is_cuda)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0)  # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(
            dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        # print(dec_outputs)
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            # print(dec_outputs)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.encoder = Encoder(args).to(device)
        self.decoder = Decoder(args).to(device)
        self.projection = nn.Linear(
            args.d_model, args.tgt_vocab_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs, keys):
        '''
        enc_inputs: [batch_size, src_lens]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, keys)
        # print(f"!!enc_self_attns: {enc_self_attns.is_cuda}")
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            dec_inputs, enc_inputs, enc_outputs, keys)
        # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


if __name__ == '__main__':
    pass
