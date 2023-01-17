import argparse

import clip
import numpy as np
import torch

from model_conversion.benchmark import benchmark_onnx, benchmark_torch
from model_conversion.utils import SIZES


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-m", "--model", type=str, help="Model name.", required=True, choices=SIZES.keys())
    arg("-n", "--num_rounds", type=int, help="The number of iterations.", default=20)
    arg("-p", "--model_path", type=str, help="Path to model in onnx format.", required=True)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    print("Loading model CPU")
    model_cpu, _ = clip.load(args.model, "cpu")

    print("Loading model GPU")
    model_gpu, _ = clip.load(args.model, "cuda")

    size = SIZES[args.model]

    dummy_input_image = torch.randn(1, 3, size, size)

    benchmark_torch(model_cpu.visual, dummy_input_image, args.num_rounds, "CPU")
    benchmark_torch(model_cpu.visual.cuda(), dummy_input_image.cuda(), args.num_rounds, "GPU")
    benchmark_torch(model_gpu.visual, dummy_input_image.cuda(), args.num_rounds, "GPU fp16", "half")
    benchmark_onnx(
        args.model_path, "CPUExecutionProvider", dummy_input_image.detach().cpu().numpy().astype(np.float32), "CPU"
    )
    benchmark_onnx(
        args.model_path, "CUDAExecutionProvider", dummy_input_image.detach().cpu().numpy().astype(np.float32), "GPU"
    )


if __name__ == "__main__":
    main()
