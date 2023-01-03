import argparse
import time

import numpy as np
import onnxruntime as ort
import torch
import transformers

from model_conversion.utils import MAX_TEXT_LENGTH, MULTILANG_MODEL_NAME, MultilingualCLIP


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-p", "--model_path", type=str, help="Path to model in onnx format.", required=True)
    return parser.parse_args()


def main() -> None:
    args = get_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(MULTILANG_MODEL_NAME)

    texts = ["Ищeм кapтинки нa Ternaus.com"]

    txt_tok = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=MAX_TEXT_LENGTH)

    model = MultilingualCLIP.from_pretrained(MULTILANG_MODEL_NAME).eval()

    model = torch.jit.trace(model, (txt_tok["input_ids"], txt_tok["attention_mask"]), strict=True)

    print("Warmup CPU")

    model_cpu = model.cpu()

    with torch.inference_mode():
        for _ in range(10):
            model_cpu(**txt_tok)

    print("Compute CPU")
    result_cpu = []

    with torch.inference_mode():
        for _ in range(10):
            start_time = time.perf_counter()
            _ = model_cpu(**txt_tok)
            result_cpu += [time.perf_counter() - start_time]

    print(
        f"Default CPU: {np.mean(result_cpu)} +- {np.std(result_cpu)}",
    )

    model_cuda = model.cuda()

    print("Warmup GPU")

    input_ids_cuda = txt_tok["input_ids"].cuda()
    attention_mask_cuda = txt_tok["attention_mask"].cuda()

    with torch.inference_mode():
        for _ in range(10):
            _ = model_cuda(input_ids_cuda, attention_mask_cuda)

    result_gpu = []

    print("Compute GPU")

    with torch.inference_mode():
        for _ in range(10):
            start_time = time.perf_counter()
            _ = model_cuda(input_ids_cuda, attention_mask_cuda)
            result_gpu += [time.perf_counter() - start_time]

    print(
        f"Default GPU: {np.mean(result_gpu)} +- {np.std(result_gpu)}",
    )

    print("Warmup GPU fp16")

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
        for _ in range(10):
            _ = model_cuda(input_ids_cuda, attention_mask_cuda)

    result_gpu_fp16 = []

    print("Compute GPU fp16")

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
        for _ in range(10):
            start_time = time.perf_counter()
            _ = model_cuda(input_ids_cuda, attention_mask_cuda)
            result_gpu_fp16 += [time.perf_counter() - start_time]

    print(
        f"Default GPU fp16: {np.mean(result_gpu_fp16)} +- {np.std(result_gpu_fp16)}",
    )

    ort_sess_cpu = ort.InferenceSession(args.model_path, providers=["CPUExecutionProvider"])

    input_ids = txt_tok["input_ids"].detach().cpu().numpy()
    attention_mask = txt_tok["attention_mask"].detach().cpu().numpy()

    result_onnx_cpu = []

    for _ in range(10):
        start_time = time.perf_counter()
        _ = ort_sess_cpu.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        result_onnx_cpu += [time.perf_counter() - start_time]

    print(
        f"Default ONNX CPU: {np.mean(result_onnx_cpu)} +- {np.std(result_onnx_cpu)}",
    )

    ort_sess_gpu = ort.InferenceSession(args.model_path, providers=["CUDAExecutionProvider"])

    result_onnx_gpu = []

    for _ in range(10):
        start_time = time.perf_counter()
        _ = ort_sess_gpu.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        result_onnx_gpu += [time.perf_counter() - start_time]

    print(
        f"Default ONNX GPU: {np.mean(result_onnx_gpu)} +- {np.std(result_onnx_gpu)}",
    )


if __name__ == "__main__":
    main()
