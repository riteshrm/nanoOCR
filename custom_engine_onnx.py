
import cv2
import json
import numpy as np
from paddleocr import TextDetection
from itertools import groupby
import cv2
import onnxruntime as ort

with open("data/processed/itoc.json", "r") as f:
    itoc = json.load(f)

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



class OCREngine:
    def __init__(self):
        self.det_model = TextDetection(
            model_name="PP-OCRv6_tiny_det",
            engine="onnxruntime",
            device="cpu",
        )
        self.rec_model = ort.InferenceSession(
            "rec_model_fp32.onnx",
            providers=["CPUExecutionProvider"]
        )

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
            out = self.rec_model.run(
                ["logits"],
                {"images": crop_img}
                )[0]
            pred_texts, _ = ctc_decoder(out.transpose(1, 0, 2))
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
