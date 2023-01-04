import time

import numpy as np
import onnxruntime as ort
import torch


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
