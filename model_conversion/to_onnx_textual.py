import argparse
import logging
from pathlib import Path
from torch import nn

import clip
import numpy as np
import onnxruntime as ort
import torch
from onnxsim import simplify

import onnx

from model_conversion.textual import Textual

SIZES = {
    "ViT-L/14": 224,
}

DEFAULT_EXPORT = dict(
    input_names=["input"],
    output_names=["output"],
    export_params=True,
    verbose=False,
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)


def onnx_checker(path: str):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    del model


def convert_visual(model: nn.Module, dummy_input: torch.Tensor, visual_path: str):
    torch.onnx.export(model.visual, dummy_input, visual_path, **DEFAULT_EXPORT)
    onnx_checker(visual_path)


def convert_textual(model: nn.Module, dummy_input: torch.Tensor, textual_path: str):
    textual = Textual(model)
    torch.onnx.export(textual, dummy_input, textual_path, **DEFAULT_EXPORT)
    onnx_checker(textual_path)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-m", "--model", type=str, help="Path to model.", required=True, choices=SIZES.keys())
    arg("-v", "--output_visual_path", type=Path, help="Path to save visual component of the model", required=True)
    arg("-t", "--output_textual_path", type=Path, help="Path to save textual component of the model", required=True)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    model, _ = clip.load(args.model, "cpu")

    visual_path = args.output_visual_path
    textual_path = args.output_textual_path

    # textual_export_params = {
    #     "input_names": ["input"],
    #     "output_names": ["output"],
    #     "export_params": True,
    #     "verbose": False,
    #     "opset_version": 17,
    #     "do_constant_folding": True,
    #     "dynamic_axes": {"input": {1: "batch_size"}, "output": {0: "batch_size"}},
    # }

    size = SIZES[args.model]

    dummy_input_image = torch.randn(1, 3, size, size)
    # convert_visual(model, dummy_input_image, visual_path)
    # visual_model_onnx = onnx.load(visual_path)  # type: ignore
    # model_simp_visual, visual_check = simplify(visual_model_onnx)
    # assert visual_check, "Simplified ONNX model could not be validated"
    # onnx.save(model_simp_visual, visual_path)  # type: ignore
    model_simp_visual = onnx.load(visual_path)  # type: ignore

    ort_sess_visual = ort.InferenceSession(model_simp_visual.SerializeToString(), providers=["CUDAExecutionProvider"])
    
    image_onnx = dummy_input_image.detach().cpu().numpy().astype(np.float32)
    input_name = ort_sess_visual.get_inputs()[0].name    
    print(input_name)
    onnx_output_visual = ort_sess_visual.run(None, {input_name: image_onnx})

    with torch.inference_mode():
        default_visual_output = model.visual(dummy_input_image)
    
    print("Visual %s ", {(default_visual_output - onnx_output_visual[0]).abs().max()})
    

    # dummy_input_text = clip.tokenize(["Search for Generated Imagert at Ternaus.com"]).cpu()  # [3, 77]
    # convert_textual(model, dummy_input_text, textual_path)
    # image_onnx = dummy_input_image.detach().cpu().numpy().astype(np.float32)

    # text_onnx = dummy_input_text.detach().cpu().numpy().astype(np.int32)

    # onnx.save(visual_model_onnx, visual_path)  # type: ignore

    # textual_model_onnx = onnx.load(textual_path)  # type: ignore
    # model_simp_textual, _ = simplify(textual_model_onnx)
    # onnx.save(textual_model_onnx, textual_path)  # type: ignore

    # with torch.inference_mode():
    #     default_visual_output = model.visual(dummy_input_image)
    #     default_textual_output = model.encode_text(dummy_input_text)

    # ort_sess_visual = ort.InferenceSession(model_simp_visual.SerializeToString(), providers=["CUDAExecutionProvider"])
    # ort_sess_textual = ort.InferenceSession(model_simp_textual.SerializeToString(), providers=["CUDAExecutionProvider"])

    # input_name = ort_sess_visual.get_inputs()[0].name
    # onnx_output_visual = ort_sess_visual.run(None, {input_name: image_onnx})

    # input_name = ort_sess_textual.get_inputs()[0].name
    # onnx_output_textual = ort_sess_textual.run(None, {input_name: text_onnx})

    # logging.info("Visual %s ", {(default_textual_output - onnx_output_textual[0]).abs().max()})
    # logging.info("Textual %s ", {(default_visual_output - onnx_output_visual[0]).abs().max()})


if __name__ == "__main__":
    main()
