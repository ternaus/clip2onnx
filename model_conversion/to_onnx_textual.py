import argparse
from pathlib import Path

import clip
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxsim import simplify
from torch import nn

from model_conversion.utils import DEFAULT_EXPORT, SIZES, Textual, onnx_checker


def convert_textual(model: nn.Module, dummy_input: torch.Tensor, textual_path: str) -> None:
    textual = Textual(model)
    torch.onnx.export(textual, dummy_input, textual_path, **DEFAULT_EXPORT)
    onnx_checker(textual_path)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-m", "--model", type=str, help="Path to model.", required=True, choices=SIZES.keys())
    arg("-o", "--output_path", type=Path, help="Path to save textual component of the model", required=True)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    model, _ = clip.load(args.model, "cpu")

    output_path = args.output_path

    dummy_input_text = clip.tokenize(["Search for Generated Imagert at Ternaus.com"]).cpu()
    convert_textual(model, dummy_input_text, output_path)

    textual_model_onnx = onnx.load(output_path)  # type: ignore

    model_simp_textual, textual_check = simplify(textual_model_onnx)

    if not textual_check:
        raise ValueError("Simplified ONNX model could not be validated")

    with torch.inference_mode():
        default_textual_output = model.encode_text(dummy_input_text)

    ort_sess_visual = ort.InferenceSession(model_simp_textual.SerializeToString(), providers=["CUDAExecutionProvider"])

    input_name = ort_sess_visual.get_inputs()[0].name

    text_onnx = dummy_input_text.detach().cpu().numpy().astype(np.int32)
    onnx_output_textual = ort_sess_visual.run(None, {input_name: text_onnx})

    print("Textual %s ", {(default_textual_output - onnx_output_textual[0]).abs().max()})


if __name__ == "__main__":
    main()
