from typing import Any

import onnx
import torch
import transformers
from multilingual_clip import Config_MCLIP
from torch import nn

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

MULTILANG_MODEL_NAME = "M-CLIP/XLM-Roberta-Large-Vit-L-14"
MAX_TEXT_LENGTH = 77


def onnx_checker(path: str) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    del model


class Textual(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.token_embedding = model.token_embedding

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x += self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # needs .float() before .argmax(  ) to work
        return x[torch.arange(x.shape[0]), text.float().argmax(dim=-1)] @ self.text_projection


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
