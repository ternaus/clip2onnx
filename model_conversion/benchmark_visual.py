import argparse
import time

import clip
import numpy as np
import onnx
import onnxruntime as ort
import torch

from model_conversion.utils import SIZES


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-m", "--model", type=str, help="Path to model.", required=True, choices=SIZES.keys())
    arg("-p", "--model_path", type=str, help="Path to model in onnx format.", required=True)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    model, _ = clip.load(args.model, "cpu")

    size = SIZES[args.model]

    dummy_input_image = torch.randn(1, 3, size, size)

    print("Warmup CPU")

    model_visual_cpu = model.visual.cpu()

    with torch.inference_mode():
        for _ in range(10):
            model_visual_cpu(dummy_input_image)

    print("Compute CPU")
    result_default_cpu = []

    with torch.inference_mode():
        for _i in range(10):
            start_time = time.perf_counter()
            _ = model_visual_cpu(dummy_input_image)
            result_default_cpu += [time.perf_counter() - start_time]

    print(
        f"Default CPU: {np.mean(result_default_cpu)} +- {np.std(result_default_cpu)}",
    )

    model_visual = model.visual.cuda()
    input_cuda = dummy_input_image.cuda()

    print("Warmup GPU")

    with torch.inference_mode():
        for _ in range(10):
            model_visual(input_cuda)

    result_default_gpu = []

    print("Compute GPU")

    with torch.inference_mode():
        for _i in range(10):
            start_time = time.perf_counter()
            _ = model_visual(input_cuda)
            result_default_gpu += [time.perf_counter() - start_time]

    print(
        f"Default GPU: {np.mean(result_default_gpu)} +- {np.std(result_default_gpu)}",
    )

    model_visual = model.visual.cuda()
    input_cuda = dummy_input_image.cuda()

    print("Warmup GPU fp16")

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
        for _ in range(10):
            model_visual(input_cuda)

    result_default_gpu = []

    print("Compute GPU fp16")

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
        for _i in range(10):
            start_time = time.perf_counter()
            _ = model_visual(input_cuda)
            result_default_gpu += [time.perf_counter() - start_time]

    print(
        f"Default GPU fp16: {np.mean(result_default_gpu)} +- {np.std(result_default_gpu)}",
    )

    visual_model_onnx = onnx.load(args.model_path)  # type: ignore
    ort_sess_visual_cpu = ort.InferenceSession(
        visual_model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    image_onnx = dummy_input_image.detach().cpu().numpy().astype(np.float32)
    input_name = ort_sess_visual_cpu.get_inputs()[0].name

    result_onnx_cpu = []

    for _ in range(10):
        start_time = time.perf_counter()
        _ = ort_sess_visual_cpu.run(None, {input_name: image_onnx})
        result_onnx_cpu += [time.perf_counter() - start_time]

    print(
        f"Default ONNX CPU: {np.mean(result_onnx_cpu)} +- {np.std(result_onnx_cpu)}",
    )

    visual_model_onnx_gpu = onnx.load(args.model_path)  # type: ignore
    ort_sess_visual_gpu = ort.InferenceSession(
        visual_model_onnx_gpu.SerializeToString(), providers=["CUDAExecutionProvider"]
    )

    image_onnx = dummy_input_image.detach().cpu().numpy().astype(np.float32)
    input_name = ort_sess_visual_gpu.get_inputs()[0].name

    result_onnx_gpu = []

    for _ in range(10):
        start_time = time.perf_counter()
        _ = ort_sess_visual_gpu.run(None, {input_name: image_onnx})
        result_onnx_gpu += [time.perf_counter() - start_time]

    print(
        f"Default ONNX GPU: {np.mean(result_onnx_gpu)} +- {np.std(result_onnx_gpu)}",
    )


if __name__ == "__main__":
    main()
