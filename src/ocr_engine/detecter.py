from paddleocr import TextDetection
det_model = TextDetection(model_name="PP-OCRv6_small_det", engine="onnxruntime", device="cpu")

