# clip2onnx
Converts CLIP models to ONNX

# Benchmark

Average over 50 iteration

## 2080Ti

* **GPU**: GTX 2080Ti
* **CPU**: AMD Ryzen Threadripper 3970X 32-Core Processor


| Mode  | Visual (seconds) | Textual (seconds) | Textual Multilang |
| ------------- | ------------- |---------|----|
| Pytorch CPU  | 0.771+-0.191  |0.068+-0.003|0.230+-0.106|
| Pytorch GPU fp32  | 0.023+-0.000 |0.004+-0.000|0.010+-0.000|
| Pytorch GPU fp16  | 0.009+-0.000 |0.004+-0.000|0.009+-0.000|
| ONNX CPU  | 0.534+-0.098  |0.042+-0.001|0.126+-0.001|
| ONNX GPU  | 0.024+-0.003  |0.005+-0.000|0.009+-0.001|
