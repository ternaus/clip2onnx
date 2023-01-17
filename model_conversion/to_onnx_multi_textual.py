import argparse
from pathlib import Path

import onnxruntime as ort
import torch
import transformers

from model_conversion.utils import MAX_TEXT_LENGTH, MULTILANG_MODEL_NAME, MultilingualCLIP


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-o", "--output_model_path", type=str, help="Path to save textual component of the model", required=True)
    arg("-t", "--output_tokenizer_path", type=Path, help="Path to save tokenizer", required=True)
    return parser.parse_args()


def main() -> None:
    args = get_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(MULTILANG_MODEL_NAME)

    torch.save(tokenizer, args.output_tokenizer_path)

    texts = ["Ищeм кapтинки нa Ternaus.com"]

    txt_tok = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=MAX_TEXT_LENGTH)

    print("Loading")
    model = MultilingualCLIP.from_pretrained(MULTILANG_MODEL_NAME).eval()

    model = torch.jit.trace(model, (txt_tok["input_ids"], txt_tok["attention_mask"]), strict=True)

    with torch.inference_mode():
        default_textual_output = model(**txt_tok)

    input_names = ["input_ids", "attention_mask"]
    output_names = ["output"]
    print("Exporting")
    torch.onnx.export(
        model,
        (txt_tok["input_ids"], txt_tok["attention_mask"]),  # texts,
        args.output_model_path,
        verbose=False,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )

    session = ort.InferenceSession(args.output_model_path, providers=["CPUExecutionProvider"])

    np_input = tokenizer(texts, return_tensors="np", padding="max_length", max_length=MAX_TEXT_LENGTH)

    onnx_output_textual = session.run(
        None, {"input_ids": np_input["input_ids"], "attention_mask": np_input["attention_mask"]}
    )[0]

    print("Textual %s ", {(default_textual_output - onnx_output_textual[0]).abs().max()})


if __name__ == "__main__":
    main()
