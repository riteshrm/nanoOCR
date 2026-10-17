from pathlib import Path
import zipfile

root = Path("det_data")

archives = {
    "ch4_training_images.zip": "train_images",
    "ch4_training_localization_transcription_gt.zip": "train_gt",
    "ch4_test_images.zip": "test_images",
    "Challenge4_Test_Task1_GT.zip": "test_gt",
}

for archive, folder in archives.items():
    with zipfile.ZipFile(root / archive) as z:
        z.extractall(root / folder)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights


class DBNet(nn.Module):
    def __init__(self, pretrained=True, channels=64, k=50):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        self.stage2 = backbone.layer1  # 64 channels,  H/4
        self.stage3 = backbone.layer2  # 128 channels, H/8
        self.stage4 = backbone.layer3  # 256 channels, H/16
        self.stage5 = backbone.layer4  # 512 channels, H/32

        # Project all feature maps to the same channel count.
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, channels, kernel_size=1)
            for c in [64, 128, 256, 512]
        ])

        # Reduce each pyramid level before concatenation.
        self.smooth = nn.ModuleList([
            nn.Conv2d(channels, channels // 4, 3, padding=1)
            for _ in range(4)
        ])

        self.prob_head = self.make_head(channels)
        self.threshold_head = self.make_head(channels)

        self.k = k

    @staticmethod
    def make_head(channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
        )

    def forward(self, x):
        output_size = x.shape[-2:]

        x = self.stem(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        features = [
            layer(feature)
            for layer, feature in zip(
                self.lateral, [c2, c3, c4, c5]
            )
        ]

        # Top-down feature fusion.
        for i in range(2, -1, -1):
            features[i] = features[i] + F.interpolate(
                features[i + 1],
                size=features[i].shape[-2:],
                mode="nearest",
            )

        target_size = features[0].shape[-2:]

        fused = torch.cat([
            F.interpolate(
                layer(feature),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            for layer, feature in zip(self.smooth, features)
        ], dim=1)

        prob_logits = F.interpolate(
            self.prob_head(fused),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        probability = torch.sigmoid(prob_logits)

        # The threshold branch is needed only during training.
        if not self.training:
            return probability

        threshold = torch.sigmoid(F.interpolate(
            self.threshold_head(fused),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ))

        binary = torch.sigmoid(
            self.k * (probability - threshold)
        )

        return {
            "prob_logits": prob_logits,
            "probability": probability,
            "threshold": threshold,
            "binary": binary,
        }


import torch.nn as nn
from pathlib import Path

import cv2
import numpy as np
import torch

from shapely.geometry import Polygon
from torch.utils.data import Dataset, DataLoader

from src.det_model.model import DBNet
def balanced_bce(logits, target, mask, negative_ratio=3):
    """
    Keep all positive pixels and the hardest negative pixels.
    Mining is performed independently for each image.
    """
    pixel_loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )

    sample_losses = []

    for loss_i, target_i, mask_i in zip(pixel_loss, target, mask):
        valid = mask_i > 0.5
        positive = (target_i > 0.5) & valid
        negative = (target_i <= 0.5) & valid

        positive_count = int(positive.sum().item())
        available_negatives = int(negative.sum().item())

        # Also support images/crops containing no text.
        negative_count = min(
            available_negatives,
            negative_ratio * max(positive_count, 1),
        )

        positive_loss = loss_i[positive].sum()

        if negative_count:
            negative_loss = loss_i[negative].topk(
                negative_count
            ).values.sum()
        else:
            negative_loss = loss_i.sum() * 0.0

        sample_losses.append(
            (positive_loss + negative_loss)
            / max(positive_count + negative_count, 1)
        )

    return torch.stack(sample_losses).mean()


def masked_l1(prediction, target, mask):
    return (
        (prediction - target).abs() * mask
    ).sum() / mask.sum().clamp_min(1.0)


def dice_loss(prediction, target, mask, eps=1e-6):
    # Sum independently over each image's channel and spatial axes.
    axes = (1, 2, 3)

    intersection = (prediction * target * mask).sum(dim=axes)
    denominator = (
        (prediction * mask).sum(dim=axes)
        + (target * mask).sum(dim=axes)
    )

    return (
        1.0 - (2.0 * intersection + eps) / (denominator + eps)
    ).mean()


class DBLoss(nn.Module):
    def forward(self, prediction, target):
        probability_loss = balanced_bce(
            prediction["prob_logits"],
            target["shrink_map"],
            target["shrink_mask"],
        )

        threshold_loss = masked_l1(
            prediction["threshold"],
            target["threshold_map"],
            target["threshold_mask"],
        )

        binary_loss = dice_loss(
            prediction["binary"],
            target["shrink_map"],
            target["shrink_mask"],
        )

        total = (
            5.0 * probability_loss
            + 10.0 * threshold_loss
            + binary_loss
        )

        return total, {
            "bce": probability_loss.detach().item(),
            "threshold": threshold_loss.detach().item(),
            "dice": binary_loss.detach().item(),
        }

def load_annotations(path):
    polygons, ignored = [], []

    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue

            parts = line.split(",", maxsplit=8)
            if len(parts) != 9:
                raise ValueError(f"Invalid annotation in {path}: {line}")

            polygon = np.asarray(
                parts[:8], dtype=np.float32
            ).reshape(4, 2)

            polygons.append(polygon)
            ignored.append(parts[8].strip() == "###")

    return polygons, ignored

def fill_geometry(canvas, geometry, value):
    """Fill a Shapely Polygon or MultiPolygon."""
    if geometry.is_empty:
        return

    if geometry.geom_type == "Polygon":
        points = np.rint(
            np.asarray(geometry.exterior.coords)
        ).astype(np.int32)

        cv2.fillPoly(canvas, [points], value)

        for hole in geometry.interiors:
            points = np.rint(
                np.asarray(hole.coords)
            ).astype(np.int32)
            cv2.fillPoly(canvas, [points], 0)

    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            fill_geometry(canvas, part, value)


def make_targets(height, width, polygons, ignored, shrink_ratio=0.4):
    shrink_map = np.zeros((height, width), dtype=np.float32)
    shrink_mask = np.ones((height, width), dtype=np.float32)

    threshold_map = np.zeros((height, width), dtype=np.float32)
    threshold_mask = np.zeros((height, width), dtype=np.float32)

    for points, ignore in zip(polygons, ignored):
        polygon = Polygon(points)

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.is_empty:
            continue

        if ignore or polygon.area < 1 or polygon.length < 1:
            fill_geometry(shrink_mask, polygon, 0)
            continue

        # DBNet's area/perimeter-based offset distance.
        distance = (
            polygon.area * (1 - shrink_ratio ** 2)
            / polygon.length
        )

        inner = polygon.buffer(-distance, join_style=1)

        # Very small or narrow polygons can disappear when shrunk.
        if inner.is_empty:
            fill_geometry(shrink_mask, polygon, 0)
            continue

        outer = polygon.buffer(distance, join_style=1)

        fill_geometry(shrink_map, inner, 1)
        fill_geometry(threshold_mask, outer, 1)

        # Rasterized polygon boundary.
        boundary = np.zeros((height, width), dtype=np.uint8)

        parts = (
            [polygon]
            if polygon.geom_type == "Polygon"
            else list(polygon.geoms)
        )

        for part in parts:
            if part.geom_type != "Polygon":
                continue

            corners = np.rint(
                np.asarray(part.exterior.coords)
            ).astype(np.int32)

            cv2.polylines(
                boundary, [corners], True, 1, thickness=1
            )

        if not boundary.any():
            continue

        # distanceTransform measures distance to zero pixels.
        distance_map = cv2.distanceTransform(
            1 - boundary,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )

        border_target = 1 - np.clip(
            distance_map / max(distance, 1e-6), 0, 1
        )

        outer_mask = np.zeros((height, width), dtype=np.float32)
        fill_geometry(outer_mask, outer, 1)

        threshold_map = np.maximum(
            threshold_map,
            border_target * outer_mask,
        )

    # Ignore regions take priority, including overlaps.
    shrink_map *= shrink_mask
    threshold_mask *= shrink_mask

    # Scale threshold supervision into [0.3, 0.7].
    threshold_map = 0.3 + 0.4 * threshold_map

    return {
        "shrink_map": torch.from_numpy(shrink_map[None]),
        "shrink_mask": torch.from_numpy(shrink_mask[None]),
        "threshold_map": torch.from_numpy(threshold_map[None]),
        "threshold_mask": torch.from_numpy(threshold_mask[None]),
    }

class DetectionDataset(Dataset):
    def __init__(self, pairs, size=512):
        self.pairs = pairs
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, annotation_path = self.pairs[index]

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        polygons, ignored = load_annotations(annotation_path)

        height, width = image.shape[:2]

        # Preserve aspect ratio, then pad to a square.
        scale = self.size / max(height, width)
        new_h = max(1, round(height * scale))
        new_w = max(1, round(width * scale))

        resized = cv2.resize(image, (new_w, new_h))

        canvas = np.full(
            (self.size, self.size, 3), 255, dtype=np.uint8
        )
        canvas[:new_h, :new_w] = resized

        # Use actual resize ratios to account for rounding.
        factors = np.array(
            [new_w / width, new_h / height],
            dtype=np.float32,
        )
        polygons = [polygon * factors for polygon in polygons]

        targets = make_targets(
            self.size, self.size, polygons, ignored
        )

        # Exclude padding from loss computation.
        valid = torch.zeros((1, self.size, self.size))
        valid[:, :new_h, :new_w] = 1

        targets["shrink_mask"] *= valid
        targets["threshold_mask"] *= valid

        image = torch.from_numpy(
            canvas.transpose(2, 0, 1).copy()
        ).float() / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        image = (image - mean) / std

        return image, targets

def get_pairs(image_dir, annotation_dir):
    pairs = []

    for image in sorted(Path(image_dir).rglob("*")):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        annotation = Path(annotation_dir) / f"gt_{image.stem}.txt"

        if not annotation.exists():
            raise FileNotFoundError(annotation)

        pairs.append((image, annotation))

    if not pairs:
        raise ValueError(f"No images found in {image_dir}")

    return pairs

torch.manual_seed(42)

image_dir = Path("det_data/train_images")
annotation_dir = Path("det_data/train_gt")

pairs = get_pairs("det_data/train_images", "det_data/train_gt")

if len(pairs) < 2:
    raise ValueError("Need at least two image/annotation pairs.")

# Split training data into train/validation.
generator = torch.Generator().manual_seed(42)
order = torch.randperm(len(pairs), generator=generator).tolist()

n_val = max(1, round(0.1 * len(pairs)))
val_pairs = [pairs[i] for i in order[:n_val]]
train_pairs = [pairs[i] for i in order[n_val:]]

train_loader = DataLoader(
    DetectionDataset(train_pairs, size=512),
    batch_size=8,
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    DetectionDataset(val_pairs, size=512),
    batch_size=8,
    shuffle=False,
    num_workers=0,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# False means the ENTIRE model starts from random weights.
model = DBNet(pretrained=False).to(device)
criterion = DBLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val = float("inf")
from tqdm import tqdm
for epoch in range(50):
    model.train()
    total_loss = 0.0
    samples = 0

    for images, targets in tqdm(train_loader):
        images = images.to(device)
        targets = {
            name: tensor.to(device)
            for name, tensor in targets.items()
        }

        prediction = model(images)
        loss, _ = criterion(prediction, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        samples += images.size(0)

    model.eval()
    val_total = 0.0
    val_samples = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            # In eval mode, DBNet returns probability only.
            probability = model(images)

            val_loss = dice_loss(
                probability,
                targets["shrink_map"].to(device),
                targets["shrink_mask"].to(device),
            )

            val_total += val_loss.item() * images.size(0)
            val_samples += images.size(0)

    train_loss = total_loss / samples
    val_loss = val_total / val_samples

    print(
        f"Epoch {epoch + 1:03d} | "
        f"train loss {train_loss:.4f} | "
        f"val Dice loss {val_loss:.4f}"
    )

    if val_loss < best_val:
        best_val = val_loss

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "val_loss": val_loss,
        }, "dbnet_best.pt")