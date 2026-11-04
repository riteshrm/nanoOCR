import json
import cv2
import numpy as np
from itertools import groupby
import onnxruntime as ort
from shapely.geometry import Polygon
import pyclipper
import math

RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
IMAGE_SHORT_SIDE=512
RESULT_DIR = 'output'
BOX_THRESH=0.7
THRESH=0.3
MIN_SIZE = 3
INFERENCE_CHECKPOINTS='inference_checkpoints'
MAX_CANDIDATES = 100

def draw_annotations(image, text_results):
    """Draw OCR boxes (top, left, bottom, right) and recognized text."""
    annotated = image.copy()
    height, width = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 180, 0)

    for index, item in enumerate(text_results, 1):
        top, left, bottom, right = map(int, item["bounding_box"])
        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left, min(right - 1, width - 1))
        bottom = max(top, min(bottom - 1, height - 1))
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)

        word = item["word"]
        if isinstance(word, list):
            word = word[0] if word else ""
        label = f"{word}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, 1
        )
        label_left = max(0, min(left, width - text_width - 6))
        label_top = top - text_height - baseline - 6
        if label_top < 0:
            label_top = min(bottom + 3, max(0, height - text_height - baseline - 6))
        cv2.rectangle(
            annotated,
            (label_left, label_top),
            (label_left + text_width + 6, label_top + text_height + baseline + 6),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            annotated, label, (label_left + 3, label_top + text_height + 3),
            font, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return annotated

#Detector
def resize_image(img):
    height, width, _ = img.shape
    if height < width:
        new_height = IMAGE_SHORT_SIDE
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = IMAGE_SHORT_SIDE
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img

def norm_resize(image):
    img = image.astype('float32')
    original_shape = img.shape[:2]
    img = resize_image(img)
    img -= RGB_MEAN
    img /= 255.
    img = img.transpose(2, 0, 1).astype(np.float32)[None, ...]
    return img, original_shape

def format_output(batch, output):
        batch_boxes, batch_scores = output
        final_boxes = []
        for index in range(batch['image'].shape[0]):
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            for i in range(boxes.shape[0]):
                score = scores[i]
                if score < BOX_THRESH:
                    continue
                box = boxes[i,:,:].reshape(-1).tolist()
                final_boxes.append(box)
        return final_boxes

def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
            points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]

def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded

def boxes_from_bitmap(pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''
        
        assert _bitmap.shape[0] == 1
        bitmap = _bitmap[0]  # The first channel
        pred = pred[0]
        height, width = bitmap.shape
        contours, _ = cv2.findContours(
            (bitmap*255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), MAX_CANDIDATES)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = get_mini_boxes(contour)
            if sside < MIN_SIZE:
                continue
            points = np.array(points)
            score = box_score_fast(pred, points.reshape(-1, 2))
            if BOX_THRESH > score:
                continue
        
            box = unclip(points).reshape(-1, 1, 2)
            box, sside = get_mini_boxes(box)
            if sside < MIN_SIZE + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()
            
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

def post_process(batch, pred):
    segmentation = (pred > THRESH).astype(int)
    boxes_batch = []
    scores_batch = []
    images = batch['image']

    for batch_index in range(images.shape[0]):
        height, width = batch['shape'][batch_index]
        boxes, scores = boxes_from_bitmap(
                pred[batch_index],
                segmentation[batch_index], width, height)
        boxes_batch.append(boxes)
        scores_batch.append(scores)
    return boxes_batch, scores_batch


#Recognizer
with open(f"{INFERENCE_CHECKPOINTS}/itoc.json", "r") as f:
    itoc = json.load(f)
itoc = {int(k): v for k, v in itoc.items()}
def process_single_sample(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 32), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = img[None, None, :, :]
    return img

def ctc_decoder(logits):
    labels = np.argmax(logits, axis=-1)
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
        decoded_indices.append(idx)
        
    return decoded_texts, decoded_indices


class DETEngine:
    def __init__(self):
        self.model = ort.InferenceSession(
                    f"{INFERENCE_CHECKPOINTS}/det_model.onnx",
                    providers=["CPUExecutionProvider"])
        
    def process(self, image):
        batch = dict()
        img, original_shape = norm_resize(image)
        batch['shape'] = [original_shape]
        batch['image'] = img
        pred = self.model.run(
                    ["pred"],
                    {"images": img})[0]
        output = post_process(batch, pred) 
        final_boxes = format_output(batch, output)
        return final_boxes


class RECEngine:
    def __init__(self):
            self.model = ort.InferenceSession(
            f"{INFERENCE_CHECKPOINTS}/rec_model.onnx",
            providers=["CPUExecutionProvider"]
            )
            
    def process(self, image):
        processed_image = process_single_sample(image)
        out = self.model.run(
        ["logits"],
        {"images": processed_image})[0]
        pred_texts, pred_labels = ctc_decoder(out.transpose(1, 0, 2))
        return pred_texts, pred_labels


class OCREngine:
    def __init__(self):
        self.det_model = DETEngine()
        self.rec_model = RECEngine()

    def process(self, image):
        """Return recognized text records and an annotated copy of the image."""
        final_boxes = self.det_model.process(image)
        
        text_results = []

        for polygon in final_boxes:
            polygon = np.asarray(polygon).astype(np.int32).reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(polygon)
            left = max(0, x)
            top = max(0, y)
            right = min(image.shape[1], x + w)
            bottom = min(image.shape[0], y + h)
            crop = image[top:bottom, left:right]
            if crop.size == 0:
                continue
            pred_texts, pred_labels = self.rec_model.process(crop)
            text_results.append({
            "word": pred_texts,
            "bounding_box": [y, x, bottom, right],})

        return text_results, draw_annotations(image, text_results)
