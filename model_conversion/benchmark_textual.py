import argparse

import clip

from model_conversion.benchmark import benchmark_onnx, benchmark_torch
from model_conversion.utils import SIZES, Textual


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-m", "--model", type=str, help="model name", required=True, choices=SIZES.keys())
    arg("-p", "--model_path", type=str, help="Path to model in onnx format.", required=True)
    arg("-n", "--num_rounds", type=int, help="The number of iterations.", default=20)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    model, _ = clip.load(args.model, "cpu")

    textual = Textual(model)
    dummy_input_text = clip.tokenize(["Search for Generated Imagert at Ternaus.com"]).cpu()

    benchmark_torch(textual, dummy_input_text, args.num_rounds, "CPU")
    benchmark_torch(textual.cuda(), dummy_input_text.cuda(), args.num_rounds, "GPU")
    benchmark_torch(textual.cuda(), dummy_input_text.cuda(), args.num_rounds, "GPU", "half")
    benchmark_onnx(args.model_path, "CPUExecutionProvider", dummy_input_text.detach().cpu().numpy(), "CPU")
    benchmark_onnx(args.model_path, "CUDAExecutionProvider", dummy_input_text.detach().cpu().numpy(), "GPU")


if __name__ == "__main__":
    main()
