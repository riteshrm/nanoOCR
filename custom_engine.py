
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from paddleocr import TextDetection, TextRecognition
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

def process_single_sample(img):

    # 3. Convert to float32 in [0, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float() / 255.0

    # 4. Resize to (32, 128)
    img = F.interpolate(img.unsqueeze(0), size=(32, 128),
                        mode="bilinear", align_corners=False).squeeze(0)

    return img

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

class OCREngine:
    def __init__(self):
        self.det_model = TextDetection(
            model_name="PP-OCRv6_tiny_det",
            engine="onnxruntime",
            device="cpu",
        )
        self.rec_model = CRNN(len(vocab)+1)

        dict = torch.load('crnn_best.pt')

        self.rec_model.load_state_dict(dict['model'])

    def process(self, image):
        det_results = self.det_model.predict(
            input=image,
            batch_size=1,
        )
        det_data = next(iter(det_results)).json["res"]

        polygons = det_data["dt_polys"]

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        text_results = []

        for polygon in polygons:
            polygon = np.asarray(polygon, dtype=np.int32)

            # Create mask for inpainting
            cv2.fillPoly(mask, [polygon], 255)

            # Axis-aligned crop for first implementation
            x, y, w, h = cv2.boundingRect(polygon)

            left = max(0, x)
            top = max(0, y)
            right = min(image.shape[1], x + w)
            bottom = min(image.shape[0], y + h)

            crop = image[top:bottom, left:right]

            if crop.size == 0:
                continue

            crop_img = process_single_sample(crop)
            out = self.rec_model(crop_img.unsqueeze(0)).permute(1, 0, 2)
            pred_texts, _ = ctc_decoder(out)
            text_results.append({
            "word": pred_texts,
            "bounding_box": [y, x, bottom, right],})

        # Expand mask slightly to cover text edges
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        redacted = cv2.inpaint(
            image,
            mask,
            inpaintRadius=3,
            flags=cv2.INPAINT_NS,
        )

        return text_results, redacted
