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


test_dataset = OCRDataset(f'{data_path}/test.csv')

from itertools import groupby

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 32
MAX_W=128

test_dataloader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, collate_fn=collate)

def ctc_decoder(probs):
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

def alignment_counts(pred, truth):
    """Return edit errors and matched characters from a minimum-edit alignment.

    Among equally short alignments, prefer the one with more exact matches.
    A substitution counts as both a false positive and a false negative.
    """
    row = [(j, 0) for j in range(len(truth) + 1)]

    for i, p in enumerate(pred, 1):
        new_row = [(i, 0)]
        for j, g in enumerate(truth, 1):
            candidates = [
                (row[j][0] + 1, row[j][1]),
                (new_row[j - 1][0] + 1, new_row[j - 1][1]),
                (row[j - 1][0] + int(p != g),
                 row[j - 1][1] + int(p == g)),
            ]
            new_row.append(min(candidates, key=lambda counts: (counts[0], -counts[1])))
        row = new_row

    return row[-1]


def evaluation_counts(predictions, targets, lengths):
    truths = torch.split(targets, lengths.tolist())
    if len(predictions) != len(truths):
        raise ValueError("Expected one prediction per ground-truth word")

    counts = {
        "errors": 0,
        "matches": 0,
        "predicted_chars": 0,
        "target_chars": 0,
        "exact_words": 0,
        "words": len(truths),
    }
    for pred, truth in zip(predictions, truths):
        pred, truth = pred.tolist(), truth.tolist()
        errors, matches = alignment_counts(pred, truth)
        counts["errors"] += errors
        counts["matches"] += matches
        counts["predicted_chars"] += len(pred)
        counts["target_chars"] += len(truth)
        counts["exact_words"] += int(pred == truth)
    return counts


model = CRNN(len(vocab)+1).to(device)
checkpoint = torch.load('crnn_best2.pt', map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model'])
model.eval()
totals = {
    "errors": 0,
    "matches": 0,
    "predicted_chars": 0,
    "target_chars": 0,
    "exact_words": 0,
    "words": 0,
}
for x, targets, target_lengths in tqdm(test_dataloader):
    x = x.to(device)
    with torch.no_grad():
        log_probs = model(x).log_softmax(-1)
    pred_texts, pred_labels = ctc_decoder(log_probs.permute(1, 0, 2))

    counts = evaluation_counts(
        pred_labels, targets, target_lengths
    )
    for name, count in counts.items():
        totals[name] += count

cer = totals["errors"] / totals["target_chars"] if totals["target_chars"] else float("nan")
# Precision is reported as zero when the model predicts no characters.
precision = totals["matches"] / totals["predicted_chars"] if totals["predicted_chars"] else 0.0
recall = totals["matches"] / totals["target_chars"] if totals["target_chars"] else 0.0
word_accuracy = totals["exact_words"] / totals["words"] if totals["words"] else 0.0
print(f"CER: {cer:.4f} ({cer:.2%})")
print(f"Character precision: {precision:.2%}")
print(f"Character recall: {recall:.2%}")
print(f"Exact word-match accuracy: {word_accuracy:.2%} "
      f"({totals['exact_words']}/{totals['words']})")
