"""Code from https://github.com/Lednik7/CLIP-ONNX."""
import torch
from torch import nn


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
