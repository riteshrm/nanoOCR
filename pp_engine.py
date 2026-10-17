
import cv2
import numpy as np
from paddleocr import TextDetection, TextRecognition


class OCREngine:
    def __init__(self):
        self.det_model = TextDetection(
            model_name="PP-OCRv6_tiny_det",
            engine="onnxruntime",
            device="cpu",
        )
        self.rec_model = TextRecognition(
            model_name="PP-OCRv6_tiny_rec",
            engine="onnxruntime",
            device="cpu",
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

            rec_results = self.rec_model.predict(
                input=crop,
                batch_size=1,
            )
            rec_data = next(iter(rec_results)).json["res"]

            text_results.append({
                "word": rec_data["rec_text"],
                "bounding_box": [y, x, bottom, right],
            })

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
