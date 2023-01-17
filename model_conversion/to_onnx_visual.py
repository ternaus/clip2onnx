import argparse
from pathlib import Path

import clip
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxsim import simplify
from torch import nn

from model_conversion.utils import DEFAULT_EXPORT, SIZES, onnx_checker


def convert_visual(model: nn.Module, dummy_input: torch.Tensor, visual_path: str) -> None:
    torch.onnx.export(model.visual, dummy_input, visual_path, **DEFAULT_EXPORT)
    onnx_checker(visual_path)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-m", "--model", type=str, help="Path to model.", required=True, choices=SIZES.keys())
    arg("-o", "--output_path", type=Path, help="Path to save visual component of the model", required=True)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    print("Loading the model")
    model, _ = clip.load(args.model, "cpu")

    output_path = args.output_path

    size = SIZES[args.model]

    dummy_input_image = torch.randn(1, 3, size, size)

    print("Converting")
    convert_visual(model, dummy_input_image, output_path)

    visual_model_onnx = onnx.load(output_path)  # type: ignore

    print("Simplifying")

    model_simp_visual, visual_check = simplify(visual_model_onnx)

    if not visual_check:
        raise ValueError("Simplified ONNX model could not be validated")

    print("Saving simplified")
    onnx.save(model_simp_visual, output_path)  # type: ignore

    ort_sess_visual = ort.InferenceSession(model_simp_visual.SerializeToString(), providers=["CUDAExecutionProvider"])

    image_onnx = dummy_input_image.detach().cpu().numpy().astype(np.float32)
    input_name = ort_sess_visual.get_inputs()[0].name

    print("Visual onnx inference")
    onnx_output_visual = ort_sess_visual.run(None, {input_name: image_onnx})

    print("Visual inference")
    with torch.inference_mode():
        default_visual_output = model.visual(dummy_input_image)

    print("Visual %s ", {(default_visual_output - onnx_output_visual[0]).abs().max()})


if __name__ == "__main__":
    main()
