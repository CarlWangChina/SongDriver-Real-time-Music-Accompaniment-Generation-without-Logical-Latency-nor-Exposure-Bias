import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
from config import Config
from torch.utils.data import DataLoader
from data import AiMusicDataset
from tqdm import tqdm
from model import Transformer

def inference(model, loader, args):


def train(model, loader, args):
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ?
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    postfix = {'running_loss': 0.0}

    for i, epoch in enumerate(range(args.epoch)):
        tqdm_train = tqdm(loader, desc='Training (epoch #{})'.format(epoch + 1))
        for batch in tqdm_train:
            enc_inputs, dec_inputs, dec_outputs = batch
            # print("size:", enc_inputs.size())
            outputs, _, _, _ = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            postfix['running_loss'] += (loss.item() - postfix['running_loss']) / (i + 1)
            tqdm_train.set_postfix(postfix)

    torch.save(model.state_dict(), 'data/model.ckpt')


if __name__ == '__main__':
    args = Config('config.yaml')
    args.input_start = 'S'
    args.output_start = 'E'

    train_dataset = AiMusicDataset('data/ori_data.txt', 'data/melody_to_id.json', 'data/chord_to_id.json', args)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn,
                                  drop_last=True)
    print('dataloader done.')
    args.src_vocab_size = len(train_dataset.src_vocab)
    args.tgt_vocab_size = len(train_dataset.tgt_vocab)

    model = Transformer(args)
    train(model, train_dataloader, args)
