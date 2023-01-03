import argparse
import time

import clip
import numpy as np
import onnx
import onnxruntime as ort
import torch

from model_conversion.textual import Textual
from model_conversion.utils import SIZES


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-m", "--model", type=str, help="model name", required=True, choices=SIZES.keys())
    arg("-p", "--model_path", type=str, help="Path to model in onnx format.", required=True)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    model, _ = clip.load(args.model, "cpu")

    textual = Textual(model)

    dummy_input_text = clip.tokenize(["Search for Generated Imagert at Ternaus.com"]).cpu()

    print("Warmup CPU")

    model_cpu = textual.cpu()

    with torch.inference_mode():
        for _ in range(10):
            model_cpu(dummy_input_text)

    print("Compute CPU")
    result_cpu = []

    with torch.inference_mode():
        for _ in range(10):
            start_time = time.perf_counter()
            _ = model_cpu(dummy_input_text)
            result_cpu += [time.perf_counter() - start_time]

    print(
        f"Default CPU: {np.mean(result_cpu)} +- {np.std(result_cpu)}",
    )

    model_cuda = textual.cuda()
    input_cuda = dummy_input_text.cuda()

    print("Warmup GPU")

    with torch.inference_mode():
        for _ in range(10):
            model_cuda(input_cuda)

    result_gpu = []

    print("Compute GPU")

    with torch.inference_mode():
        for _ in range(10):
            start_time = time.perf_counter()
            _ = model_cuda(input_cuda)
            result_gpu += [time.perf_counter() - start_time]

    print(
        f"Default GPU: {np.mean(result_gpu)} +- {np.std(result_gpu)}",
    )

    print("Warmup GPU fp16")

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
        for _ in range(10):
            model_cuda(input_cuda)

    result_gpu_fp16 = []

    print("Compute GPU fp16")

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
        for _ in range(10):
            start_time = time.perf_counter()
            _ = model_cuda(input_cuda)
            result_gpu_fp16 += [time.perf_counter() - start_time]

    print(
        f"Default GPU fp16: {np.mean(result_gpu_fp16)} +- {np.std(result_gpu_fp16)}",
    )

    model_onnx_cpu = onnx.load(args.model_path)  # type: ignore
    ort_sess_cpu = ort.InferenceSession(model_onnx_cpu.SerializeToString(), providers=["CPUExecutionProvider"])

    text_onnx = dummy_input_text.detach().cpu().numpy()
    input_name = ort_sess_cpu.get_inputs()[0].name

    result_onnx_cpu = []

    for _ in range(10):
        start_time = time.perf_counter()
        _ = ort_sess_cpu.run(None, {input_name: text_onnx})
        result_onnx_cpu += [time.perf_counter() - start_time]

    print(
        f"Default ONNX CPU: {np.mean(result_onnx_cpu)} +- {np.std(result_onnx_cpu)}",
    )

    model_onnx_gpu = onnx.load(args.model_path)  # type: ignore
    ort_sess_gpu = ort.InferenceSession(model_onnx_gpu.SerializeToString(), providers=["CUDAExecutionProvider"])

    text_onnx = dummy_input_text.detach().cpu().numpy()
    input_name = ort_sess_gpu.get_inputs()[0].name

    result_onnx_gpu = []

    for _ in range(10):
        start_time = time.perf_counter()
        _ = ort_sess_gpu.run(None, {input_name: text_onnx})
        result_onnx_gpu += [time.perf_counter() - start_time]

    print(
        f"Default ONNX GPU: {np.mean(result_onnx_gpu)} +- {np.std(result_onnx_gpu)}",
    )


if __name__ == "__main__":
    main()
