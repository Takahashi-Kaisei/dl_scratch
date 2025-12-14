import torch
from torch import nn


class CausalAttention(nn.Module):
    """
    maskありのattention
    """

    def __init__(self, d: int):
        super().__init__()
        self.scale = d ** (1 / 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        """
        Args:
            q: (seq_length, embedding_size)
            k: (seq_length, embedding_size)
            v: (seq_length, embedding_size)
        """
        seq_len = q.shape[-2]  # 後ろから二番目がsequense lengthに対応する。
        score = (q @ k.mT) / self.scale
        # q.deviceを参照することで、GPU/CPU問わず動作する堅牢なコードになります
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device), diagonal=1
        ).bool()
        score = score.masked_fill(mask, -torch.inf)

        return self.softmax(score) @ v


class MultiHeadAttention(nn.Module):
    def __init__(
        self, n_head, d_model
    ):  # ここは64じゃなくてn_headの方が良い．契約プログラミング的に．64とすると全体で見た時マジックナンバーっぽくなる．
        super().__init__()
        assert d_model % n_head == 0, "d_modelがn_headで割り切れません。"
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = CausalAttention(d_model)

    def forward(self, q, k, v):
        qw = self.w_q(q)
        kw = self.w_k(k)
        vw = self.w_v(v)
        calculated = []
        # chunkでhead数に分割する．
        for qw_i, kw_i, vw_i in zip(
            qw.chunk(self.n_head, dim=-1),
            kw.chunk(self.n_head, dim=-1),
            vw.chunk(self.n_head, dim=-1),
        ):
            calculated.append(self.attention(qw_i, kw_i, vw_i))
        return self.w_o(torch.cat(calculated, dim=-1))
