# clip2onnx
Converts CLIP models to ONNX

# Convert models to ONNX
## Visual
`python -m model_conversion.to_onnx_visual -m "ViT-L/14" -o v.onnx`

## Textual
`python -m model_conversion.to_onnx_textual -m "ViT-L/14" -o t.onnx`

## Multi langual textual
`python -m model_conversion.to_onnx_multi_textual -o mt.onnx -t t.pt`

# Benchmark
`python -m model_conversion.benchmark_visual -m "ViT-L/14" -p v.onnx -n 50`
`python -m model_conversion.benchmark_textual -m "ViT-L/14" -p t.onnx -n 50`


Average over 50 iteration

## 2080Ti

* **GPU**: GTX 2080 Ti
* **CPU**: AMD Ryzen Threadripper 3970X 32-Core Processor


| Mode  | Visual (seconds) | Textual (seconds) | Textual Multilang |
| ------------- | ------------- |---------|----|
| Pytorch CPU  | 0.771+-0.191  |0.068+-0.003|0.230+-0.106|
| Pytorch GPU fp32  | 0.023+-0.000 |0.004+-0.000|0.010+-0.000|
| Pytorch GPU fp16  | 0.009+-0.000 |0.004+-0.000|0.009+-0.000|
| ONNX CPU  | 0.534+-0.098  |0.042+-0.001|0.126+-0.001|
| ONNX GPU  | 0.024+-0.003  |0.005+-0.000|0.009+-0.001|

## 1080 Ti

* **GPU**: GTX 1080 Ti
* **CPU**: Intel(R) Core(TM) i3-9350KF CPU @ 4.00GHz

| Mode  | Visual (seconds) | Textual (seconds) | Textual Multilang |
| ------------- | ------------- |---------|----|
| Pytorch CPU  |0.503+-0.004|0.051+- 0.000|0.191+-0.000|
| Pytorch GPU fp32  |0.037+-0.000|0.008+-0.000|0.017+-0.000|
| Pytorch GPU fp16  |0.036+-0.000|0.006+-0.000|0.019+-0.000|
| ONNX CPU  |0.715+-0.003|0.065+-0.000|0.221+-0.001|
| ONNX GPU  ||||

## A4000

* **GPU**: A4000
* **CPU**: AMD EPYC 7551P 32-Core Processor

| Mode  | Visual (seconds) | Textual (seconds) | Textual Multilang |
| ------------- | ------------- |---------|----|
| Pytorch CPU  |13.428+-3.554|0.669+-0.286|3.362+-1.183|
| Pytorch GPU fp32  |0.038+-0.011|0.021+-0.008|0.031+-0.017|
| Pytorch GPU fp16  |0.061+-0.022|0.028+-0.009|0.087+-0.045|
| ONNX CPU  |4.070+-1.286|0.364+-0.116|3.906+-2.229|
| ONNX GPU  ||||