"""Load actor MLP weights from ``*.pt`` checkpoints (no Isaac / rsl_rl import)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _extract_checkpoint_dict(ckpt: Any) -> dict[str, Any]:
    if isinstance(ckpt, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        return ckpt
    raise TypeError(f"Unsupported checkpoint type {type(ckpt)}")


def _actor_weight_keys(sd: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for k in sd:
        if "critic" in k.lower():
            continue
        if "actor" not in k.lower() and "policy" not in k.lower():
            continue
        if k.endswith(".weight") and isinstance(sd[k], torch.Tensor) and sd[k].dim() == 2:
            keys.append(k)
    return keys


def _layer_order(key: str) -> int:
    m = re.search(r"\.(\d+)\.weight$", key)
    return int(m.group(1)) if m else 10_000


def _linear_stack(sd: dict[str, Any], weight_keys: list[str]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, wk in enumerate(sorted(weight_keys, key=_layer_order)):
        w = sd[wk]
        bias_key = wk.replace(".weight", ".bias")
        bias = sd.get(bias_key)
        if bias is None:
            raise KeyError(f"Missing bias for {wk}")
        lin = nn.Linear(w.shape[1], w.shape[0], bias=True)
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(bias)
        layers.append(lin)
        if i < len(weight_keys) - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def load_actor_mlp(path: str | Path, device: torch.device) -> nn.Module:
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = _extract_checkpoint_dict(ckpt)
    wkeys = _actor_weight_keys(sd)
    if not wkeys:
        raise KeyError(f"No actor/policy *.weight keys in {path}")
    net = _linear_stack(sd, wkeys).to(device)
    net.eval()
    return net


@torch.no_grad()
def infer_actions(net: nn.Module, obs_451: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.as_tensor(obs_451, dtype=torch.float32, device=device).unsqueeze(0)
    return net(x).squeeze(0).cpu().numpy()
