import argparse

import torch
import transformers

from model_conversion.benchmark import benchmark_onnx, benchmark_torch
from model_conversion.utils import MAX_TEXT_LENGTH, MULTILANG_MODEL_NAME, MultilingualCLIP


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-p", "--model_path", type=str, help="Path to model in onnx format.", required=True)
    arg("-n", "--num_rounds", type=int, help="The number of iterations.", default=20)
    return parser.parse_args()


def main() -> None:
    args = get_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(MULTILANG_MODEL_NAME)

    texts = ["Ищeм кapтинки нa Ternaus.com"]

    txt_tok = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=MAX_TEXT_LENGTH)

    model = MultilingualCLIP.from_pretrained(MULTILANG_MODEL_NAME).eval()

    model = torch.jit.trace(model, (txt_tok["input_ids"], txt_tok["attention_mask"]), strict=True)

    # benchmark_torch(
    #     model, {"input_ids": txt_tok["input_ids"], "attention_mask": txt_tok["attention_mask"]}, args.num_rounds, "CPU"
    # )
    benchmark_torch(
        model.cuda(),
        {"input_ids": txt_tok["input_ids"].cuda(), "attention_mask": txt_tok["attention_mask"].cuda()},
        args.num_rounds,
        "GPU",
    )
    benchmark_torch(
        model.cuda(),
        {"input_ids": txt_tok["input_ids"].cuda(), "attention_mask": txt_tok["attention_mask"].cuda()},
        args.num_rounds,
        "GPU fp16",
        "half",
    )

    benchmark_onnx(
        args.model_path,
        "CPUExecutionProvider",
        {"input_ids": txt_tok["input_ids"].numpy(), "attention_mask": txt_tok["attention_mask"].numpy()},
        "CPU",
    )

    benchmark_onnx(
        args.model_path,
        "CUDAExecutionProvider",
        {"input_ids": txt_tok["input_ids"].numpy(), "attention_mask": txt_tok["attention_mask"].numpy()},
        "GPU",
    )


if __name__ == "__main__":
    main()
