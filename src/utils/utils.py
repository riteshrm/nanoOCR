import torch
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
from itertools import groupby
import os
import shutil
from pathlib import Path
def cleanup_checkpoints(ckpt_dir, prefix, keep=3):
    checkpoints = [
        d for d in os.listdir(ckpt_dir)
        if d.startswith(prefix)
        and os.path.isdir(os.path.join(ckpt_dir, d))
    ]

    checkpoints.sort(
        key=lambda x: int(x.rsplit("-", 1)[-1]),
        reverse=True
    )

    for ckpt in checkpoints[keep:]:
        shutil.rmtree(os.path.join(ckpt_dir, ckpt))


def get_latest_checkpoint(ckpt_dir, prefix="checkpoint-"):
    checkpoints = [
        p for p in Path(ckpt_dir).glob(f"{prefix}*")
        if p.is_dir() and p.name[len(prefix):].isdigit()
    ]

    if not checkpoints:
        return None, 0

    latest = max(
        checkpoints,
        key=lambda p: int(p.name[len(prefix):])
    )

    epoch = int(latest.name[len(prefix):])

    return latest, epoch

def process_single_sample(img_path, label):
    img = read_image(img_path, mode=ImageReadMode.GRAY)
    img = img.float() / 255.0
    img = F.interpolate(img.unsqueeze(0), size=(32, 128),
                        mode="bilinear", align_corners=False).squeeze(0)

    return img,label

def encode(txt, ctoi):
    dig_lst = []
    
    for char in (txt):
        if char in ctoi:
            dig_lst.append(ctoi.get(char))
    return torch.tensor(dig_lst, dtype=torch.long)

def ctc_decoder(probs, itoc):
        labels = torch.argmax(probs, dim=-1)
        decoded_texts = []
        decoded_indices = []
        for i in range(labels.shape[0]):
            ans = ""
            idx = []
        
            merged_list = [k for k,_ in groupby(labels[i])]
            
            for p in merged_list:
                if p != 0:
                    ans += itoc[int(p)]
                    idx.append(int(p))

            
            decoded_texts.append(ans)
            decoded_indices.append(torch.tensor(idx, dtype=torch.int64))
            
        return decoded_texts, decoded_indices

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