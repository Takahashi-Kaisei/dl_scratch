# %%
from torch import nn
from .attention import MultiHeadAttention


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class TransformerLayer(nn.Module):
    """transformerlayer入力形状と同じものが返る。

    Ex:
        model = TransformerLayer(8, 512, 2048)
        x = torch.tensor(torch.rand(32, 9, 512), dtype=torch.float32)
        result = model(x)
        print(result)
    """

    def __init__(self, n_head, d_model, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(n_head, d_model)
        self.ffn = FFN(d_model, d_ff)
        self.layer_norm_1 = nn.LayerNorm(
            d_model
        )  # normレイヤーはパラメータを共有しているわけでは無いので，二つインスタンスが必要．
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):  # (batch, seq_len, d_model)
        residual_1 = x  # 接続用に保存しておく．
        x = self.mha(x, x, x)
        x = x + residual_1  # 残差結合
        x = self.layer_norm_1(x)  # post norm
        residual_2 = x
        x = self.ffn(x)
        x = x + residual_2
        x = self.layer_norm_2(x)
        return x
