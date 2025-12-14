import torch
from torch import nn
from .layers import TransformerLayer


class Transformer(nn.Module):

    def __init__(
        self, vocab_size, n_head, d_model, d_ff, n_layers, context_window
    ):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )
        self.transformer_layers = nn.Sequential(
            *[TransformerLayer(n_head, d_model, d_ff) for _ in range(n_layers)]
        )  # いけてる？
        self.linear = nn.Linear(
            d_model, vocab_size
        )  # batch_size, seq_len, vocab_sizeになる
        self.softmax = nn.Softmax(dim=-1)
        # NOTE: ここの初期化方法。
        self.pe = nn.Parameter(
            torch.randn(
                [context_window, d_model]
            )  # ここvocab_sizeじゃなくてd_modelじゃね？
        )  # 上限までベクトルを作って，forwardで削れば良い．

    def forward(self, x):  # batch_size, seq_len: int
        x = self.embed(x)  # batch_size, seq_len, d_model
        x = x + self.pe[: x.size(-2), :]  # batch_size, seq_len, d_model
        x = self.transformer_layers(x)  # batch_size, seq_len, d_model
        x = self.linear(x)  # batch_size, seq_len, vocab_size
        return x  # batch_size, seq_len, vocab_size
        # return self.softmax(x) # batch_size, seq_len, vocab_size
