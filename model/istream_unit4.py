import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import Config
from data import AiMusicDataset
from model import Transformer
from tqdm import tqdm


def queueShift(que, n=1):  # 移动队列
    l = len(que)
    if l > n:
        for i in range(l-n):
            que[i] = que[i+n]


class processor:
    def __init__(self, args, m, tgt_vocab, idx2word):
        self.model = Transformer(args)
        self.model.load_state_dict(m)
        self.processIndex = 16
        self.idx2word = idx2word
        self.tgt_vocab = tgt_vocab
        self.start_symbol = tgt_vocab["S"]
        self.dec_input = torch.zeros(
            1, args.tgt_len).type_as(torch.LongTensor())
        self.model.eval()

    def encode(self, input):  # 编码器
        self.enc_outputs, enc_self_attns = self.model.encoder(input)
        self.enc_input = input

    def decode(self):  # 解码器
        dec_outputs, _, _ = self.model.decoder(
            self.dec_input, self.enc_input, self.enc_outputs)
        projected = self.model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        return prob.data[self.processIndex]

    def step(self):  # 执行一步操作，上下文留在self.dec_input里面
        queueShift(self.dec_input[0])  # 移动队列
        res = self.decode()  # 执行解码器
        self.dec_input[0][self.processIndex] = res
        predict, _, _, _ = self.model(
            self.enc_input[0].view(1, -1), self.dec_input)
        predict = predict.data.max(1, keepdim=True)[1]
        return self.idx2word[predict.squeeze()[self.processIndex].item()]


# 测试
if __name__ == '__main__':
    # 加载配置
    args = Config('config.yaml')
    args.input_start = 'S'
    args.output_start = 'E'
    args.batch_size = 4
    # 加载数据
    test_dataset = AiMusicDataset(
        'data/demo_dataset.txt', 'data/melody_to_id.json', 'data/chord_to_id.json', args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=test_dataset.collate_fn,
                                 drop_last=False)
    idx2word = test_dataset.idx2tgt
    tgt_vocab = test_dataset.tgt_vocab
    print('dataloader done.')
    args.src_vocab_size = len(test_dataset.src_vocab)
    args.tgt_vocab_size = len(test_dataset.tgt_vocab)
    # 初始化模型
    p = processor(args, torch.load('data/model.ckpt'), tgt_vocab, idx2word)
    for batch in tqdm(test_dataloader):
        enc_inputs, _, _ = batch
        print("Encoder_input_size:", enc_inputs.size())
        for i in range (0, enc_inputs.size()[1], 4):
            enc_inputs_concat = torch.zeros((1, 32), dtype=torch.long) # 4个音符一组 变成长度为32的数组
            for j in range (0, 32):
                enc_inputs_concat[0][j] = enc_inputs[0][i + j%4]
            print(f'batch, iter:{i}, enc_inputs:{enc_inputs_concat}')
            # 旋律进入
            p.encode(enc_inputs_concat[0].view(1, -1))
            # step函数执行输出
            for i in range(8):
                pass
                # print(p.step())
