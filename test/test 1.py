import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import Config
from data import AiMusicDataset
from model import Transformer
from tqdm import tqdm
import chordPreprocess

'''
用法：
export PYTHONPATH=.
python3.9 test.py
'''

# 修改以下参数
model_dir = './data' # 存放model的文件夹路径
testset_path = './dataset/test_set.txt'
test_range = (-1, 30) # 填写预测checkpoint的id 如从model_0.ckpt到model_99.ckpt 就写(0,99)
precision = 1  # 预测使用ckpt的间隔数 设置为10即每10个ckpt预测意思 全部预测就写1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def greedy_decoder(model, enc_input, start_symbol, keys, args):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input, keys)
    dec_input = torch.zeros(1, args.tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, args.tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs, keys)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
        # print(next_symbol)
    # print(dec_input)
    return dec_input


def test(model, loader, idx2word, tgt_vocab, args):
    global ckpt, model_dir
    model.eval()
    pred_result = []
    single_line = ''
    for batch in tqdm(loader):
        keys, enc_inputs, _, _ = batch
        keys = keys.to(device)
        enc_inputs =  enc_inputs.to(device)

        src_dict_path = './data/melody_to_id.json'
        with open(src_dict_path) as f:
            src_dict = json.load(f)
        id_to_melody = {v:k for k,v in src_dict.items()}

        melody = id_to_melody[int(enc_inputs[0][0])]
        melody_list = [int(id_to_melody[int(a)]) for a in list(list(enc_inputs[0].view(1, -1))[0])]

        mel_arr = []
        for it in enc_inputs[0]:
            mel_arr.append(id_to_melody[int(it)])
        chordPreprocess.pushMelody(mel_arr)

        greedy_dec_input = greedy_decoder(model, enc_inputs[0].view(1, -1), start_symbol=tgt_vocab["S"], keys=keys, args=args)
        greedy_dec_input = greedy_dec_input.to(device)
        predict, _, _, _ = model(enc_inputs[0].view(1, -1), greedy_dec_input, keys)
        predict = predict.data.max(1, keepdim=True)[1]
        # print(enc_inputs[0], '->', [idx2word[n.item()] for n in predict.squeeze()])

        tgt_dict_path = './data/chord_to_id.json'
        with open(tgt_dict_path) as f:
            tgt_dict = json.load(f)
        f.close()
        id_to_chord = {v:k for k,v in tgt_dict.items()}

        chord = [idx2word[n.item()] for n in predict.squeeze()]
        if chord[0].isdigit():
            # print(chord)
            chord_list = [int(chord[0][i]+chord[0][i+1]) for i in range(0, len(chord[0]), 2)]
            # print(chord_list)
            single_line += f'{melody_list}|{chord_list}\n'
            chordPreprocess.pushChord(chord[0])
        else:
            single_line += f'{melody_list}|{chord}\n'
            chordPreprocess.pushChord('000000')

        save_txt_path = f'{model_dir}/results/test_ckpt{ckpt}.txt'
        with open(save_txt_path, 'w') as f:
            f.write(single_line)

        pred_result.append([idx2word[n.item()] for n in predict.squeeze()])

    # pred_result = np.concatenate(pred_result, axis=-1)
    # np.save(f'model200/npy/pred_result_{ckpt}.npy', pred_result)


def batch_test ():
    global ckpt
    print(f"ckpt{ckpt}")
    args = Config('config.yaml')
    args.input_start = 'S'
    args.output_start = 'E'
    args.batch_size = 1

    test_dataset = AiMusicDataset(f'{testset_path}', 'data/melody_to_id.json', 'data/chord_to_id.json', args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=test_dataset.collate_fn,
                                 drop_last=False)
    idx2word = test_dataset.idx2tgt
    tgt_vocab = test_dataset.tgt_vocab
    print('dataloader done.')
    args.src_vocab_size = len(test_dataset.src_vocab)
    args.tgt_vocab_size = len(test_dataset.tgt_vocab)
    model = Transformer(args)
    model.to(device)
    
    model.load_state_dict(torch.load(f'{model_dir}/model_{ckpt}.ckpt', map_location='cpu'))
    print('model done.')

    test(model, test_dataloader, idx2word, tgt_vocab, args)


if __name__ == '__main__':
    if not os.path.exists(f'{model_dir}/results'):
        os.makedirs(f'{model_dir}/results')

    ckpt = test_range[1]
    while ckpt > test_range[0]:
        batch_test()
        ckpt -= precision