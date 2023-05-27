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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, loader, args):
    global device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ?
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    postfix = {'running_loss': 0.0}

    for i, epoch in enumerate(range(args.epoch)):
        tqdm_train = tqdm(loader, desc='Training (epoch #{})'.format(epoch + 1))
        for batch in tqdm_train:
            keys, enc_inputs, dec_inputs, dec_outputs = batch
            keys = keys.to(device)
            for key in enc_inputs.keys():
                enc_inputs[key] = enc_inputs[key].to(device)
            for key in dec_inputs.keys():
                dec_inputs[key] = dec_inputs[key].to(device)
            dec_outputs = dec_outputs.to(device)
            outputs, _, _, _ = model(enc_inputs, dec_inputs, keys)
            outputs = outputs.to(device)
            loss = criterion(outputs, dec_outputs.view(-1))
            loss = loss.to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            postfix['running_loss'] += (loss.item() - postfix['running_loss']) / (i + 1)
            tqdm_train.set_postfix(postfix)
            # print(enc_inputs[0], dec_inputs[0])

        torch.save(model.state_dict(), f'data/model_{i}.ckpt')


if __name__ == '__main__':
    args = Config('config.yaml')
    args.input_start = 'S'
    args.output_start = 'E'
    print(f"batch_size:{args.batch_size}")

    train_dataset = AiMusicDataset('dataset/all-half.pre.4.txt', 'data/melody_to_id.json', 'data/chord_to_id.json', args)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn,
                                  drop_last=True)
    print('dataloader done.')
    args.src_vocab_size = len(train_dataset.src_vocab)
    args.tgt_vocab_size = len(train_dataset.tgt_vocab)

    model = Transformer(args)
    # model.load_state_dict(torch.load(f'./ckpt250/model_200.ckpt', map_location='cpu'))
    model = model.to(device)
    train(model.to(device), train_dataloader, args)
