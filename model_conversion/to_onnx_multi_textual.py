import argparse
from pathlib import Path
from typing import Any

import onnxruntime as ort
import torch
import transformers
from multilingual_clip import Config_MCLIP
from torch import nn

MAX_TEXT_LENGTH = 77

model_name = "M-CLIP/XLM-Roberta-Large-Vit-L-14"


class MultilingualCLIP(transformers.PreTrainedModel):
    config_class = Config_MCLIP.MCLIPConfig

    def __init__(self, config: Config_MCLIP.MCLIPConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.transformer = transformers.AutoModel.from_pretrained(config.modelBase)
        self.LinearTransformation = torch.nn.Linear(
            in_features=config.transformerDimensions, out_features=config.numDims
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        txt_tok_new = {"input_ids": input_ids, "attention_mask": attention_mask}
        embs = self.transformer(**txt_tok_new)[0]
        att = txt_tok_new["attention_mask"]
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.LinearTransformation(embs)

    @classmethod
    def _load_state_dict_into_model(
        cls, model: nn.Module, state_dict: dict, pretrained_model_name_or_path: str, _fast_init: bool = True
    ) -> tuple[nn.Module, list[str], list[str], list[str]]:
        model.load_state_dict(state_dict)
        return model, [], [], []


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-o", "--output_model_path", type=str, help="Path to save textual component of the model", required=True)
    arg("-t", "--output_tokenizer_path", type=Path, help="Path to save tokenizer", required=True)
    return parser.parse_args()


def main() -> None:
    args = get_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    torch.save(tokenizer, args.output_tokenizer_path)

    texts = ["Ищeм кapтинки нa Ternaus.com"]

    txt_tok = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=MAX_TEXT_LENGTH)

    model = MultilingualCLIP.from_pretrained(model_name).eval()

    model = torch.jit.trace(model, (txt_tok["input_ids"], txt_tok["attention_mask"]), strict=True)

    with torch.inference_mode():
        default_textual_output = model(**txt_tok)

    input_names = ["input_ids", "attention_mask"]
    output_names = ["output"]

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
