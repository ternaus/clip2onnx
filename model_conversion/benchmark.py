import time

import numpy as np
import onnxruntime as ort
import torch


def prediction_torch(model: torch.nn.Module, x: dict[str, torch.Tensor] | torch.Tensor) -> None:
    model(**x) if isinstance(x, dict) else model(x)


def benchmark_torch(
    model: torch.nn.Module,
    x: dict[str, torch.Tensor] | torch.Tensor | np.ndarray,
    num_rounds: int,
    device: str,
    mode: str = "full",
) -> None:

    print()

    if mode == "half":
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            if isinstance(x, torch.Tensor):
                x = x.half()
            for _ in range(num_rounds):
                prediction_torch(model, x)

            print(f"Compute {device}")
            result = []

            for _ in range(num_rounds):
                start_time = time.perf_counter()
                prediction_torch(model, x)
                result += [time.perf_counter() - start_time]
    elif mode == "full":
        with torch.inference_mode():
            for _ in range(num_rounds):
                prediction_torch(model, x)

            print(f"Compute {device}")
            result = []

            for _ in range(num_rounds):
                start_time = time.perf_counter()
                prediction_torch(model, x)
                result += [time.perf_counter() - start_time]

    print(f"Result {device}: {np.mean(result): .3f}+-{np.std(result): .3f}")


def prediction_onnx(sesssion: ort.InferenceSession, x: dict[str, np.ndarray] | np.ndarray) -> None:
    sesssion.run(None, x) if isinstance(x, dict) else sesssion.run(None, {"input": x})


def benchmark_onnx(
    model_path: str, provider: str, input_array: dict[str, np.ndarray] | np.ndarray, device: str
) -> None:
    print()
    ort_sess = ort.InferenceSession(model_path, providers=[provider])

    result = []

    for _ in range(10):
        start_time = time.perf_counter()
        prediction_onnx(ort_sess, input_array)
        result += [time.perf_counter() - start_time]

    print(
        f"ONNX {device}: {np.mean(result): .3f}+-{np.std(result): .3f}",
    )
