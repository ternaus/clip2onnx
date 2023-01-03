import argparse
import time

import clip
import numpy as np
import onnxruntime as ort
import torch

from model_conversion.utils import SIZES


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-m", "--model", type=str, help="Model name.", required=True, choices=SIZES.keys())
    arg("-n", "--num_rounds", type=int, help="The number of iterations.", default=20)
    arg("-p", "--model_path", type=str, help="Path to model in onnx format.", required=True)
    return parser.parse_args()


def benchmark_onnx(model_path: str, provider: str, input_array: np.ndarray, device: str) -> None:
    ort_sess = ort.InferenceSession(model_path, providers=[provider])

    input_name = ort_sess.get_inputs()[0].name

    result = []

    for _ in range(10):
        start_time = time.perf_counter()
        _ = ort_sess.run(None, {input_name: input_array})
        result += [time.perf_counter() - start_time]

    print(
        f"Default ONNX {device}: {np.mean(result): .3f}+-{np.std(result): .3f}",
    )


def benchmark_torch(
    model: torch.nn.Module, input_array: torch.Tensor, num_rounds: int, device: str, mode: str = "full"
) -> None:

    print()

    if mode == "half":
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            for _ in range(num_rounds):
                model(input_array)

            print(f"Compute {device}")
            result = []

            for _ in range(num_rounds):
                start_time = time.perf_counter()
                _ = model(input_array)
                result += [time.perf_counter() - start_time]
    elif mode == "full":
        with torch.inference_mode():
            for _ in range(num_rounds):
                model(input_array)

            print(f"Compute {device}")
            result = []

            for _ in range(num_rounds):
                start_time = time.perf_counter()
                _ = model(input_array)
                result += [time.perf_counter() - start_time]

    print(f"Result {device}: {np.mean(result): .3f}+-{np.std(result): .3f}")


def main() -> None:
    args = get_args()
    print("Load model")
    model, _ = clip.load(args.model, "cpu")

    size = SIZES[args.model]

    dummy_input_image = torch.randn(1, 3, size, size)

    benchmark_torch(model.visual, dummy_input_image, args.num_rounds, "CPU")
    benchmark_torch(model.visual.cuda(), dummy_input_image.cuda(), args.num_rounds, "GPU")
    benchmark_torch(model.visual.cuda(), dummy_input_image.cuda(), args.num_rounds, "GPU", "half")
    benchmark_onnx(
        args.model_path, "CPUExecutionProvider", dummy_input_image.detach().cpu().numpy().astype(np.float32), "CPU"
    )
    benchmark_onnx(
        args.model_path, "CUDAExecutionProvider", dummy_input_image.detach().cpu().numpy().astype(np.float32), "GPU"
    )


if __name__ == "__main__":
    main()
