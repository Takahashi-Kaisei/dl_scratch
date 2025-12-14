# %%
import sys
from pathlib import Path
import torch

sys.path.append(f"{Path(__file__).resolve().parents[1]}/modules")
from transformer_scratch.attention import CausalAttention

x = torch.tensor(torch.rand(32, 9, 512), dtype=torch.float32)

d_model = 512
n_head = 8
d_head = d_model / n_head
attention = CausalAttention(d=d_head)
result = attention(x, x, x)
assert result.size() == x.size(), "failed"
print("success")
