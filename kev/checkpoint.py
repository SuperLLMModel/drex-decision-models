# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Jared Palmer
# Adapted from Kev (https://github.com/jaredpalmer/kev).
# Modified by Nace.AI for local merged Drex checkpoints and pointer heads.
# See NOTICE and docs/upstream.md for attribution and revision information.
"""Load a merged Drex checkpoint.

The backbone shards live next to ``head.pt``. There is no adapter to merge.
"""
from pathlib import Path
import pickle

import torch

from .model import DecisionModel, load_tokenizer


def read_meta(path):
    path = Path(path)
    try:
        meta = torch.load(path / "head.pt", map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError("head.pt must contain only tensors and primitive metadata; unsafe pickle objects are rejected") from exc
    if not isinstance(meta, dict) or not isinstance(meta.get("head"), dict):
        raise ValueError("head.pt must be a metadata dictionary containing a head state dictionary")
    return meta


def load(path, device=None, dtype=None):
    """Return ``(tokenizer, model)`` in eval mode.

    ``device`` defaults to CUDA, then Apple MPS, then CPU. ``dtype`` defaults
    to bfloat16 on CUDA and MPS, and float32 on CPU.
    """
    path = Path(path)
    meta = read_meta(path)
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    tok = load_tokenizer(str(path))
    model = DecisionModel(
        str(path),
        tok,
        device,
        lora=None,
        head_dim=int(meta.get("head_dim", 256)),
        option_isolation=bool(meta.get("option_isolation", False)),
        dtype=dtype,
        state_bidir=bool(meta.get("state_bidir", True)),
    )
    model.head.load_state_dict(meta["head"])
    model.head.temperature = float(meta.get("temperature", 1.0))
    model.eval()
    return tok, model
