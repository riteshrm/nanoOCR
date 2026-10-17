import torch
import torch.nn as nn
import torch.nn.functional as F
data_path = "data/processed"
from itertools import groupby
import pandas as pd
import cv2
traindata = pd.read_csv(f'{data_path}/train.csv')
valdata = pd.read_csv(f'{data_path}/val.csv')
testdata = pd.read_csv(f'{data_path}/test.csv')

alltexts = list(traindata['chars'])

max_label_len = max([len(x) for x in alltexts])

vocab_set = set(''.join(alltexts))
vocab_set.add('j')
vocab = sorted(vocab_set)

ctoi = {c: i+1 for i, c in enumerate(vocab)}
itoc = {i: c for c, i in ctoi.items()}
def ctc_decoder(logits):
    probs = F.softmax(logits, dim=-1)
    labels = torch.argmax(probs, dim=-1)
    decoded_texts = []
    decoded_indices = []
    for i in range(labels.shape[0]):
        ans = ""
        idx = []
        
        ## merge repeats
        merged_list = [k for k,_ in groupby(labels[i])]
        
        ## remove blanks
        for p in merged_list:
            if p != 0:
                ans += itoc[int(p)]
                idx.append(int(p))

        
        decoded_texts.append(ans)
        decoded_indices.append(torch.tensor(idx, dtype=torch.int64))
        
    return decoded_texts, decoded_indices


H, W = 32, 128

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, padding=1, pool_size=(2, 2)):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool2d(pool_size)
    def forward(self, x):
        x = self.cnn(x)
        x = self.batchnorm(x)
        x = self.act(x)
        x = self.maxpool(x)
        return x


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        channels = [ 1, 64, 128, 256, 256]
        pooling = [(2, 2), (2, 2), (2, 1), (2, 1)]
        self.cnns = nn.ModuleList([CNNBlock(in_channels=channels[i], out_channels=channels[i+1], pool_size=pooling[i]) for i, pool_size in enumerate(pooling)])
        self.rnn1 = nn.LSTM(256 * (H // 16), 128, num_layers=2,
                           bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(256, 128, num_layers=2,
                                   bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, num_classes)
 
    def forward(self, x):
        for cnnblock in self.cnns:
            x = cnnblock(x)
        b, c, h, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * h)   # (B, T, C*h)
        
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        return self.fc(x).permute(1, 0, 2)