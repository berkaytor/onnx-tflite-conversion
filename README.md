# ONNX to TFLite Conversion

This project focuses on converting an ONNX model to TFLite, tested using the Sapiens depth model.

## Prerequisites
- Python 3.8
- Install dependencies from `requirements.txt`

## Getting the ONNX Model from PyTorch

1. Clone the Sapiens-PyTorch-Inference repository:
   ```sh
   git clone https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference.git
   cd Sapiens-Pytorch-Inference
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the following script to generate the ONNX model from the PyTorch model:
   ```sh
   python onnx_export.py depth03b
   ```

## Converting ONNX to TFLite

1. Clone this conversion repository:
   ```sh
   git clone https://github.com/berkaytor/onnx-tflite-conversion.git
   cd onnx-tflite-conversion
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Ensure you have `model.onnx` in your working directory.

### Step 1: Convert ONNX to TensorFlow `.pb` File
Run the following script to generate the TensorFlow protobuf file:
   ```sh
   python convert.py
   ```

### Step 2: Convert TensorFlow Model to TFLite
Run the following script to generate the TFLite model:
   ```sh
   python tflite.py
   ```

### Step 3: Verify the Results
1. Ensure `sample.png` (input) is in the directory.
2. Run the verification script:
   ```sh
   python verify.py
   ```

![](sample.jpg)


## Notes
- Ensure that all dependencies are installed properly before running scripts.
- The ONNX model must be available in the working directory before conversion.

## References
- [Sapiens-PyTorch-Inference](https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference)

