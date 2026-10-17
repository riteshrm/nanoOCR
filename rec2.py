import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import trackio
from accelerate import Accelerator
from torchvision.io import read_image, ImageReadMode
from rec_model_1 import CRNN
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


def run(args):
    train_dataset = OCRDataset(f'{data_path}/train.csv')
    val_dataset = OCRDataset(f'{data_path}/val.csv')

    from itertools import groupby

    device = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH, EPOCHS, LR = 32, 200, 1e-3
    MAX_W=128

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH,shuffle=True, collate_fn=collate)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH, shuffle=True, collate_fn=collate)

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
    model = CRNN(len(vocab)+1, H).to(device)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)  # ignores samples whose label is too long for T
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(train_dataloader))

    best_cer = float("inf")
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
        total_errors = total_chars = 0
        total_val_loss=0
        with torch.no_grad():
            for x, targets, target_lengths in val_dataloader:
                x = x.to(device)
                log_probs = model(x).log_softmax(-1)
                input_lengths = torch.tensor([log_probs.shape[0]] * log_probs.shape[1], dtype=torch.int64)
                val_loss = ctc(log_probs, targets, input_lengths, target_lengths)
                pred_texts, pred_labels = ctc_decoder(log_probs.permute(1, 0, 2))
                print("Predictions:", [repr(p) for p in pred_texts[:10]])
                print("Blank fraction:",
                    (log_probs.argmax(-1) == 0).float().mean().item())

                errors, chars = cer_counts(
                    pred_labels, targets, target_lengths
                )
                total_errors += errors
                total_chars += chars
                total_val_loss+=val_loss.item()

        cer = total_errors / total_chars
        print(f"epoch {epoch + 1:02d}  train loss {total / len(train_dataloader):.3f}  val loss {total_val_loss / len(val_dataloader):.3f} val CER {cer:.3f}")
        if cer < best_cer :
            best_cer  = cer
            torch.save({"model": model.state_dict(), "chars": vocab, "H": H, "MAX_W": MAX_W},
                        "crnn_best2.pt")
        
def main():
    parser = argparse.ArgumentParser(description="Process raw text to tokens")
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--mlp_type", type=str, default='ffn')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--validation_step", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_data_path", type=str, default='data/regex_pretrain_train.npy')
    parser.add_argument("--val_data_path", type=str, default='data/regex_pretrain_val.npy')

    args = parser.parse_args()
    
    run(args=args)


if __name__ == "__main__":
    main()