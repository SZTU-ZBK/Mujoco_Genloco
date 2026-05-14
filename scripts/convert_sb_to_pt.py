#!/usr/bin/env python3
"""Convert stable-baselines GenLoco .zip policy to PyTorch .pt for infer/policy_mlp.py."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import torch


def convert_sb_zip_to_pt(zip_path: Path, out_path: Path) -> None:
    """Extract actor MLP weights from a stable-baselines PPO .zip and save as PyTorch .pt."""

    z = zipfile.ZipFile(zip_path)
    npz = np.load(z.open("parameters"))

    # Read hyperparameters to verify architecture
    data = json.loads(z.read("data").decode("utf-8"))
    net_arch = data.get("policy_kwargs", {}).get("net_arch", [{"pi": [64, 64], "vf": [64, 64]}])
    print("net_arch:", net_arch)

    # Build torch state dict with keys that policy_mlp.py will recognise
    sd: dict[str, torch.Tensor] = {}

    # Actor layers: pi_fc0 -> actor.0, pi_fc1 -> actor.2, pi -> actor.4
    # TF dense kernel shape is [in, out]; PyTorch Linear weight is [out, in]
    actor_map = {
        "actor.0.weight": "model/pi_fc0/w:0",
        "actor.0.bias": "model/pi_fc0/b:0",
        "actor.2.weight": "model/pi_fc1/w:0",
        "actor.2.bias": "model/pi_fc1/b:0",
        "actor.4.weight": "model/pi/w:0",
        "actor.4.bias": "model/pi/b:0",
    }

    for pt_key, sb_key in actor_map.items():
        arr = npz[sb_key]
        if pt_key.endswith(".weight"):
            # Transpose from TF [in, out] to PyTorch [out, in]
            arr = arr.T
        sd[pt_key] = torch.from_numpy(arr)
        print(f"  {pt_key}: {sd[pt_key].shape}")

    # Optional: also store critic for completeness (not used by inference)
    critic_map = {
        "critic.0.weight": "model/vf_fc0/w:0",
        "critic.0.bias": "model/vf_fc0/b:0",
        "critic.2.weight": "model/vf_fc1/w:0",
        "critic.2.bias": "model/vf_fc1/b:0",
        "critic.4.weight": "model/vf/w:0",
        "critic.4.bias": "model/vf/b:0",
    }
    for pt_key, sb_key in critic_map.items():
        arr = npz[sb_key]
        if pt_key.endswith(".weight"):
            arr = arr.T
        sd[pt_key] = torch.from_numpy(arr)

    torch.save({"state_dict": sd}, out_path)
    print(f"Saved PyTorch checkpoint to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", type=Path, required=True, help="Input stable-baselines .zip")
    ap.add_argument("--out", type=Path, required=True, help="Output .pt path")
    args = ap.parse_args()
    convert_sb_zip_to_pt(args.zip, args.out)


if __name__ == "__main__":
    main()
