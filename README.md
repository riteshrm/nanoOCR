
# nanoOCR

A lightweight, end-to-end OCR pipeline for text detection, recognition, and image annotation, optimized for CPU inference using ONNX Runtime and INT8 quantization.

## Project Overview

nanoOCR consists of two deep learning models:

1. **Text Detector (DBNet):** Identifies text regions within an image.
2. **Text Recognizer (CNN-BiLSTM):** Recognizes text from detected regions using CTC decoding.

Both the command-line interface (CLI) and FastAPI-based web application produce annotated images with text bounding boxes and recognized text labels, along with JSON results.

## Quick Start: Inference

The inference engine uses the INT8 ONNX versions of both the detection and recognition models.

Create and activate the inference environment, then install the dependencies:

```bash
conda create -n inference_env_name python=3.10 -y
conda activate inference_env_name
pip install -r infer_requirements.txt
```

You can run either the web app or the CLI.

### Web App

```bash
uvicorn main:app
```

The web app previews and downloads an annotated image with green bounding boxes
and text labels, plus the recognized text as JSON. Bounding boxes in
the JSON use `[top, left, bottom, right]` coordinates. Labels use OpenCV's basic
Latin font; labels may overlap in dense images.

### CLI

```bash
./textrecognizer image.jpg
```

This saves two files:

- `image_annotated.png`: Shows the original image with text bounding boxes and recognized text labels.
- `image_text.json`: Contains the text present in the image.

## Example Results

The following example shows the original image alongside its annotated output, with bounding boxes and recognized text labels.

| Input image | Annotated image |
| --- | --- |
| ![Input image](assets/img.jpg) | ![Annotated image with text bounding boxes and labels](assets/img_annotated.png) |

To reproduce this example:

```bash
python cli.py assets/img.jpg --output-dir assets
```

**Detected text and bounding boxes:**

- **Wellington:** `[325, 404, 395, 857]`
- **Park:** `[314, 877, 394, 1085]`

See the [recognized text and bounding boxes](assets/img_text.json) for the full output.


## Models and Datasets

### Models

1. The detection module is adapted from [DBNet](https://github.com/MhLiao/DB).
2. The recognition module is adapted from [Spark AI Summit 2020 Text Extraction](https://github.com/rajesh-bhat/spark-ai-summit-2020-text-extraction).

### Datasets

1. IIIT5K
2. MJSynth for text recognition.
3. TD TR for text region detection.
4. TextOCR for text region detection.

## Architecture Overview

### Detector Module

The detector module is adapted from the DBNet repository and consists of four modules:

1. **Feature extractor (ResNet18):** Extracts features from the input image.
2. **Feature unifier:** Combines features from multiple ResNet18 stages using convolutional layers.
3. **Probability predictor (`binarize`):** Predicts the probability that each pixel belongs to a text region.
4. **Threshold predictor (`thresh`):** Predicts a threshold for each pixel. This threshold map is combined with the probability map to produce an approximate binary map during training.

The training loss is a weighted sum of L1 loss for the threshold map, Dice loss for the approximate binary map, and balanced binary cross-entropy (BCE) loss for the per-pixel text probabilities.

The detector is evaluated on the held-out TDTR test set using precision, recall, and H-mean.

### Recognizer Module

The recognizer consists of two modules:

1. **Feature extractor (CNN):** Extracts features from text regions.
2. **Decoder (BiLSTM):** Processes the extracted feature sequence to predict characters.

The recognizer is trained using Connectionist Temporal Classification (CTC) loss and evaluated on held-out test data using character error rate (CER), exact word-match accuracy, character precision, and character recall.

## Dataset Preparation

Create and activate the training environment, then install the dependencies:

```bash
conda create -n train_env_name python=3.10 -y
conda activate train_env_name
pip install -r train_requirements.txt
```

### Detection Model

1. Download the [TextOCR data](https://textvqa.org/textocr/dataset/).
2. Place all files in the project root directory and rename the JSON files to `train.json`, `val.json`, and `test.json`, respectively.
3. Download the [TD TR data](https://paddleocr.bj.bcebos.com/dataset/TD_TR.tar) and place it in the same root directory.
4. Run the following command from the project root directory:

```bash
python -m src.data.det_data_processing
```

This processes the data and creates three folders in `det_data/`:

- `processed_train`
- `processed_val`
- `processed_test`

### Recognition Model

1. Download the [MJSynth data](https://www.robots.ox.ac.uk/~vgg/data/text/).
2. Place the file in the project root directory.
3. Run the following command from the project root directory:

```bash
python -m src.data.rec_data_processing
```

You can specify how many samples to use from MJSynth. The script automatically downloads the IIIT5K dataset from Kaggle.

This processes the data and creates multiple CSV files in `rec_data/processed/`, including train, validation, and test CSV files, as well as separate CSV files for both datasets.

## Training

### Detection Model

```bash
python -m src.det_model.train
```

You can specify `batch_size`. Checkpoints are saved in `det_checkpoints/` in the project root directory, with the prefixes `best-ckpt-` and `ckpt-`. The best checkpoint is selected based on the H-mean score.

During training, the model is also evaluated on the validation data, reporting precision, recall, and H-mean.

### Recognition Model

```bash
python -m src.rec_model.train
```

You can specify `batch_size`. Checkpoints are saved in `rec_checkpoints/` in the project root directory, with the prefixes `best-ckpt-` and `ckpt-`. The best checkpoint is selected based on the character error rate (CER).

During training, the model is also evaluated on the validation data, reporting CER.

## Exporting Models

To export the detection model, run:

```bash
python -m src.det_model.export
```

This saves FP32 and INT8 models as `det_model_fp32.onnx` and `det_model_int8.onnx`, along with the optimized INT8 model `det_model.onnx`, in `inference_checkpoints/`.

To export the recognition model, run:

```bash
python -m src.rec_model.export
```

This saves FP32 and INT8 models as `rec_model_fp32.onnx` and `rec_model_int8.onnx`, along with the optimized INT8 model `rec_model.onnx`, in `inference_checkpoints/`.

## Evaluation

Each model has separate evaluation scripts for PyTorch and ONNX models.

### Detection Model

To evaluate the PyTorch detector on the test data, run:

```bash
python -m src.eval.detector
```

To evaluate the ONNX detector (INT8 or FP32), run:

```bash
python -m src.eval.onnx_detector
```

### Recognition Model

To evaluate the PyTorch recognizer on the test data, run:

```bash
python -m src.eval.recognizer
```

To evaluate the ONNX recognizer (INT8 or FP32), run:

```bash
python -m src.eval.onnx_recognizer
```

## Evaluation Results

All evaluation metrics and latency measurements below were measured on CPU, using the same evaluation datasets for each model across PyTorch FP32, ONNX Runtime FP32, and ONNX Runtime INT8.

- **Environment:** Ubuntu on WSL2
- **CPU:** Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
- **CPU threads:** 4
- **Batch size:** 1
- **Warmup:** 20 forward passes
- **Latency measurement:** Forward pass only, excluding preprocessing and postprocessing.

### Detection Model

Results on the TDTR test set:

| Model format | Precision | Recall | H-mean | Mean latency (ms) | Median latency (ms) | P95 latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| INT8 ONNX | 0.293 | 0.363 | 0.324 | 214.704 | 214.029 | 247.256 |
| FP32 ONNX | 0.296 | 0.373 | 0.330 | 224.976 | 220.783 | 261.635 |
| FP32 PyTorch | 0.296 | 0.373 | 0.330 | 290.916 | 285.088 | 326.251 |

### Recognition Model

Results on the IIIT5K test set:

| Model format | CER | Character precision | Character recall | Exact word-match accuracy | Mean latency (ms) | Median latency (ms) | P95 latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| INT8 ONNX | 0.1429 (14.29%) | 87.72% | 87.54% | 68.90% (2067/3000) | 2.453 | 2.412 | 3.078 |
| FP32 ONNX | 0.1441 (14.41%) | 87.47% | 87.45% | 68.90% (2067/3000) | 3.030 | 3.022 | 3.975 |
| FP32 PyTorch | 0.1441 (14.41%) | 87.47% | 87.45% | 68.90% (2067/3000) | 6.146 | 5.717 | 8.185 |

## CPU Inference Optimization

Both models were exported from PyTorch to ONNX in FP32 and quantized to INT8. Moving from PyTorch FP32 to ONNX Runtime FP32 improved inference latency. Moving from ONNX Runtime FP32 to INT8 provided a smaller but still measurable improvement for both models.

Speedups based on median latency:

| Model | ONNX FP32 vs. PyTorch FP32 | ONNX INT8 vs. ONNX FP32 | ONNX INT8 vs. PyTorch FP32 |
| --- | --- | --- | --- |
| Detector | 1.29× | 1.03× | 1.33× |
| Recognizer | 1.89× | 1.25× | 2.37× |

## Limitations and Future Work

The current implementation has several areas for improvement:

1. **Detection accuracy:** The detector currently achieves an H-mean of approximately 0.33, which limits overall text detection performance. Further training could improve performance.
2. **Recognition accuracy:** The recognizer achieves approximately 68.9% exact word-match accuracy on the IIIT5K test set. Further training could improve performance.
3. **Model architecture:** The initial goal was to establish a working baseline. Lightweight backbones such as LCNetV4 from PPOCRv6 and EfficientNet-inspired architectures have not yet been explored and could improve the trade-off between accuracy and latency.
4. **Training data:** Increasing the training sample size improved generalization for both models. Additional training data, augmentation, and hyperparameter tuning could improve it further.
5. **Quantization:** INT8 quantization reduced mean latency, with a small decrease in detection accuracy and a small improvement in recognition CER. Further profiling could identify bottlenecks and help guide the exploration of other quantization strategies.

## AI Usage

I have used AI tools primarily for technical discussions, debugging, code review, documentation preparation, and web interface development.
