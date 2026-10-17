"""1D-CNN beat classifier (Full / Lite) with RR features concatenated after GAP.

  4 x [Conv1d(same, no bias) -> BN -> ReLU], MaxPool(2) + Dropout(p_conv) after blocks 1-3
  -> global average pooling -> concat standardised RR (4) -> FC1 -> ReLU -> Dropout(p_head) -> FC2 (4 classes)

Full: channels 64/128/256/512, kernels 7/5/3/3, head 516 -> 256 -> 4 (518 with 6 RR features).
Lite: channels 16/32/64/64,   kernels 7/5/3/3, head 68 -> 32 -> 4  (70 with 6 RR features).
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn

N_RR = 4
N_CLASSES = 4


@dataclass(frozen=True)
class ModelConfig:
    channels: tuple[int, ...]
    kernels: tuple[int, ...]
    hidden: int
    dropout_conv: float = 0.2
    dropout_head: float = 0.5
    n_rr: int = N_RR                    # 4, or 6 with the extended RR features (A4)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(tuple(d["channels"]), tuple(d["kernels"]), int(d["hidden"]),
                   float(d["dropout_conv"]), float(d["dropout_head"]), int(d.get("n_rr", N_RR)))

    def to_dict(self) -> dict:
        return asdict(self)


FULL = ModelConfig((64, 128, 256, 512), (7, 5, 3, 3), 256)
LITE = ModelConfig((16, 32, 64, 64), (7, 5, 3, 3), 32)


class ECGCNN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if len(cfg.channels) != 4 or len(cfg.kernels) != 4:
            raise ValueError("expected 4 conv blocks")
        layers, c_in = [], 1
        for i, (c, k) in enumerate(zip(cfg.channels, cfg.kernels)):
            layers += [nn.Conv1d(c_in, c, k, padding=k // 2, bias=False), nn.BatchNorm1d(c), nn.ReLU()]
            if i < 3:
                layers += [nn.MaxPool1d(2), nn.Dropout(cfg.dropout_conv)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(c_in + cfg.n_rr, cfg.hidden)
        self.drop = nn.Dropout(cfg.dropout_head)
        self.fc2 = nn.Linear(cfg.hidden, N_CLASSES)
        # RR standardisation (inner-train mu/sigma of the fold), stored in the checkpoint.
        # For the C export these are folded into FC1: W' = W / sigma, b' = b - W @ (mu / sigma).
        self.register_buffer("rr_mean", torch.zeros(cfg.n_rr))
        self.register_buffer("rr_std", torch.ones(cfg.n_rr))

    def set_rr_stats(self, mean, std) -> None:
        self.rr_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.rr_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        h = self.gap(self.features(x)).squeeze(-1)                 # (B, C_last)
        rr = (rr - self.rr_mean) / self.rr_std
        h = torch.relu(self.fc1(torch.cat([h, rr], dim=1)))
        return self.fc2(self.drop(h))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
