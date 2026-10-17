import argparse
import json

import cv2
import numpy as np
import torch

from shapely.geometry import Polygon

from src.det_model.model import DBNet  # Change this import to your model file.


def preprocess(image, size=512):
    """BGR image -> normalized RGB tensor, preserving training preprocessing."""
    height, width = image.shape[:2]

    scale = size / max(height, width)
    new_h = max(1, round(height * scale))
    new_w = max(1, round(width * scale))

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (new_w, new_h))

    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    tensor = torch.from_numpy(
        canvas.transpose(2, 0, 1).copy()
    ).float() / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0), new_h, new_w


def polygon_score(probability, contour):
    """Mean probability inside a contour."""
    x, y, width, height = cv2.boundingRect(contour)

    local_contour = contour.copy()
    local_contour[:, 0, 0] -= x
    local_contour[:, 0, 1] -= y

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [local_contour], 1)

    region = probability[y:y + height, x:x + width]
    return float(cv2.mean(region, mask=mask)[0])


def extract_boxes(
    probability,
    original_shape,
    binary_threshold=0.3,
    box_threshold=0.5,
    unclip_ratio=1.5,
    min_size=3,
):
    """
    probability: [resized_height, resized_width], excluding padding.
    Returns polygons in original image coordinates.
    """
    resized_h, resized_w = probability.shape
    original_h, original_w = original_shape[:2]

    bitmap = (probability > binary_threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        bitmap,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    results = []

    for contour in contours:
        if len(contour) < 3:
            continue

        rect = cv2.minAreaRect(contour)
        if min(rect[1]) < min_size:
            continue

        score = polygon_score(probability, contour)
        if score < box_threshold:
            continue

        # The model predicts shrunken regions. Expand them before cropping.
        polygon = Polygon(cv2.boxPoints(rect))

        if not polygon.is_valid or polygon.area <= 0:
            continue

        distance = (
            polygon.area * unclip_ratio
            / max(polygon.length, 1e-6)
        )
        expanded = polygon.buffer(distance, join_style=1)

        if expanded.is_empty or expanded.geom_type != "Polygon":
            continue

        points = np.asarray(
            expanded.exterior.coords, dtype=np.float32
        )

        box = cv2.boxPoints(cv2.minAreaRect(points))

        # Map from resized image to original image.
        box[:, 0] *= original_w / resized_w
        box[:, 1] *= original_h / resized_h

        box[:, 0] = np.clip(box[:, 0], 0, original_w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, original_h - 1)

        box = np.rint(box).astype(np.int32)

        left, top = box.min(axis=0)
        right, bottom = box.max(axis=0)

        if right <= left or bottom <= top:
            continue

        results.append({
            "polygon": box.tolist(),
            "bounding_box": [
                int(top), int(left), int(bottom), int(right)
            ],
            "score": score,
        })

    # Simple top-to-bottom ordering, not full document reading order.
    results.sort(key=lambda result: (
        result["bounding_box"][0],
        result["bounding_box"][1],
    ))

    return results


@torch.inference_mode()
def detect(
    model,
    image,
    device,
    size=512,
    binary_threshold=0.3,
    box_threshold=0.5,
    unclip_ratio=1.5,
):
    tensor, new_h, new_w = preprocess(image, size)

    # model.eval() makes the earlier DBNet return probability only.
    probability = model(tensor.to(device))[0, 0].cpu().numpy()

    # Remove padded area before postprocessing.
    probability = probability[:new_h, :new_w]

    results = extract_boxes(
        probability,
        image.shape,
        binary_threshold=binary_threshold,
        box_threshold=box_threshold,
        unclip_ratio=unclip_ratio,
    )

    return results, probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="dbnet_best.pt")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--binary_threshold", type=float, default=0.3)
    parser.add_argument("--box_threshold", type=float, default=0.5)
    parser.add_argument("--unclip_ratio", type=float, default=1.5)
    parser.add_argument("--output", default="detected.png")
    parser.add_argument("--json_output", default="detections.json")
    parser.add_argument("--map_output", default="probability.png")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Must use the same architecture/channels as training.
    model = DBNet(pretrained=False)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()

    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Cannot read image: {args.image}")

    results, probability = detect(
        model,
        image,
        device,
        size=args.size,
        binary_threshold=args.binary_threshold,
        box_threshold=args.box_threshold,
        unclip_ratio=args.unclip_ratio,
    )

    annotated = image.copy()

    for result in results:
        polygon = np.asarray(result["polygon"], dtype=np.int32)
        cv2.polylines(
            annotated, [polygon], True, (0, 255, 0), 2
        )

    # Save probability map at original image dimensions.
    probability_image = cv2.resize(
        probability,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    probability_image = np.clip(
        probability_image * 255, 0, 255
    ).astype(np.uint8)

    for path, output in [
        (args.output, annotated),
        (args.map_output, probability_image),
    ]:
        if not cv2.imwrite(path, output):
            raise OSError(f"Cannot save image: {path}")

    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Detected {len(results)} text regions")
    print(f"Boxes: {args.output}")
    print(f"Probability map: {args.map_output}")
    print(f"JSON: {args.json_output}")


if __name__ == "__main__":
    main()