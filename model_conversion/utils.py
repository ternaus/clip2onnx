import onnx

SIZES = {
    "ViT-L/14": 224,
}

DEFAULT_EXPORT = {
    "input_names": ["input"],
    "output_names": ["output"],
    "export_params": True,
    "verbose": False,
    "opset_version": 17,
    "do_constant_folding": True,
    "dynamic_axes": {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
}


def onnx_checker(path: str) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    del model
