import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode

data_path = "data/processed"

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

import torch.nn.functional as F
import torch

def encode(txt):
    dig_lst = []
    
    for char in (txt):
        if char in ctoi:
            dig_lst.append(ctoi.get(char))
    return torch.tensor(dig_lst, dtype=torch.long)



def process_single_sample(img_path, label):
    # 1-2. Read + decode as grayscale -> uint8, shape [1, H, W]
    img = read_image(img_path, mode=ImageReadMode.GRAY)

    # 3. Convert to float32 in [0, 1]
    img = img.float() / 255.0

    # 4. Resize to (32, 128)
    img = F.interpolate(img.unsqueeze(0), size=(32, 128),
                        mode="bilinear", align_corners=False).squeeze(0)

    return img,label

from torch.utils.data import Dataset, DataLoader

class OCRDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        self.data_csv = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        image_path, text = self.data_csv.iloc[idx]
        return process_single_sample(image_path, encode(text))

def collate(batch):
    images, labels = zip(*batch)
    target_lengths = torch.tensor(
        [len(label) for label in labels], dtype=torch.long
    )
    return torch.stack(images), torch.cat(labels), target_lengths

H, W = 32, 128
import torch.nn as nn
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


train_dataset = OCRDataset(f'{data_path}/train.csv')
val_dataset = OCRDataset(f'{data_path}/val.csv')

from itertools import groupby

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH, EPOCHS, LR = 256, 30, 1e-3
MAX_W=128

train_dataloader = DataLoader(train_dataset, batch_size=BATCH,shuffle=True, collate_fn=collate)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH, shuffle=True, collate_fn=collate)

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

from tqdm import tqdm

def eval(pred, gt, lens):
    lens = lens.tolist()
    bsz = len(pred)
    total_count=sum(lens)
    correct_count = 0
    start=0
    for i in range(bsz):
        for pl, gl in zip(pred[i], gt[start:start+lens[i]]):
            if torch.equal(pl, gl):
                correct_count+=1
        start = start+lens[i]
    return total_count, correct_count


def edit_distance(pred, truth):
    row = list(range(len(truth) + 1))

    for i, p in enumerate(pred, 1):
        new_row = [i]
        for j, g in enumerate(truth, 1):
            new_row.append(min(
                row[j] + 1,                  # delete predicted character
                new_row[j - 1] + 1,          # insert missing character
                row[j - 1] + int(p != g),    # substitute
            ))
        row = new_row

    return row[-1]


def cer_counts(predictions, targets, lengths):
    truths = torch.split(targets, lengths.tolist())
    errors = sum(
        edit_distance(pred.tolist(), truth.tolist())
        for pred, truth in zip(predictions, truths)
    )
    return errors, sum(lengths.tolist())
model = CRNN(len(vocab)+1).to(device)
ctc = nn.CTCLoss(blank=0, zero_infinity=True)  # ignores samples whose label is too long for T
opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(train_dataloader))

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for x, targets, target_lengths in tqdm(train_dataloader):
        x = x.to(device)
        log_probs = model(x).log_softmax(-1)
        input_lengths = torch.tensor([log_probs.shape[0]] * log_probs.shape[1], dtype=torch.int64)
        loss = ctc(log_probs, targets, input_lengths, target_lengths)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        total += loss.item()

    model.eval()
    correct = n = 0
    with torch.no_grad():
        for x, targets, target_lengths in tqdm(val_dataloader):
            pred_texts, pred_labels = ctc_decoder(model(x.to(device)).permute(1, 0, 2))
            total_count, correct_count = eval(pred_labels, targets, target_lengths)
            correct += correct_count
            n += total_count
    acc = correct / n
    print(f"epoch {epoch + 1:02d}  loss {total / len(train_dataloader):.3f}  val exact-match {acc:.3f}")
    if acc >= best_acc:
        best_acc = acc
        torch.save({"model": model.state_dict(), "chars": vocab, "H": H, "MAX_W": MAX_W},
                    "crnn_best.pt")