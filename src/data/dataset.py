import os
import torch
from torch.utils.data import Dataset
from src.utils.utils import process_single_sample, encode
import pandas as pd
class RECDataset(Dataset):
    def __init__(self, data_path, csv_name):
        super().__init__()
        self.data_csv = pd.read_csv(os.path.join(data_path, csv_name))
        self.ctoi = torch.load(os.path.join(data_path, 'vocab.pt'))['ctoi']

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        image_path, text = self.data_csv.iloc[idx]
        return process_single_sample(image_path, encode(text, self.ctoi))

def rec_collate(batch):
    images, labels = zip(*batch)
    target_lengths = torch.tensor(
        [len(label) for label in labels], dtype=torch.long
    )
    return torch.stack(images), torch.cat(labels), target_lengths

class DETDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data_files = os.listdir(self.data_path)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.data_path, self.data_files[idx]), weights_only=False)
        return data['image'], data['gt'], data['mask'], data['thresh_map'], data['thresh_mask']

def det_collate(batch):
    images, gts, masks, thresh_maps, thresh_masks = zip(*batch)
    images, gts, masks, thresh_maps, thresh_masks = torch.stack(images), torch.stack(gts), torch.stack(masks), torch.stack(thresh_maps), torch.stack(thresh_masks)
    final_data = {'image':images, 'gt':gts, 'mask':masks, 'thresh_map':thresh_maps, 'thresh_mask':thresh_masks}
    return final_data

class DETValDataset(DETDataset):
    def __getitem__(self, idx):
        path = os.path.join(self.data_path, self.data_files[idx])
        data = torch.load(path, map_location="cpu", weights_only=False)

        return {
            "image": data["image"],
            "polygons": data["polygons"],
            "ignore_tags": data["ignore_tags"],
            "filename": data["filename"],
            "shape": tuple(data["shape"]),
        }

def det_val_collate(samples):
    return {
        "image": torch.stack([s["image"] for s in samples]),
        "polygons": [s["polygons"] for s in samples],
        "ignore_tags": [s["ignore_tags"] for s in samples],
        "filename": [s["filename"] for s in samples],
        "shape": [s["shape"] for s in samples],
    }