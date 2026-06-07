"""Delay-pattern codebook utilities + codec-vocab special ids.

Ported verbatim from sgl-project/sglang-omni ``higgs_tts/utils.py`` (Apache-2.0).

Codebook ``c`` is shifted later in time by ``c`` steps. The leading ``c`` rows of
codebook ``c`` are padded with ``BOC_ID`` and trailing rows with ``EOC_ID``. These
ids live *inside* the per-codebook vocab (size 1026 = 1024 acoustic codes + BOC + EOC),
NOT in the text vocab.
"""

from __future__ import annotations

import torch

# Codec-vocab specials (inside the [N*V] codebook space, NOT the text vocab).
BOC_ID = 1024  # beginning-of-codes pad
EOC_ID = 1025  # end-of-codes pad / stop signal on codebook 0


def apply_delay_pattern(codes_TN: torch.Tensor) -> torch.Tensor:
    """``[T, N]`` raw codes -> ``[T + N - 1, N]`` delayed, BOC/EOC padded.

    Used to delay reference-audio codes before they are spliced onto the
    ``-100`` placeholders for voice cloning.
    """
    if codes_TN.ndim != 2:
        raise ValueError(
            f"codes_TN must be 2-D [T, N], got shape {tuple(codes_TN.shape)}"
        )
    T, N = codes_TN.shape
    out = torch.full(
        (T + N - 1, N), EOC_ID, device=codes_TN.device, dtype=codes_TN.dtype
    )
    t_idx = torch.arange(T + N - 1, device=codes_TN.device)
    for c in range(N):
        out[t_idx < c, c] = BOC_ID
        out[c : c + T, c] = codes_TN[:, c]
    return out


def reverse_delay_pattern(delayed_LN: torch.Tensor) -> torch.Tensor:
    """``[L, N]`` delayed (L >= N) -> ``[L - (N - 1), N]`` raw codes.

    Used to un-delay the generated audio tokens before codec decode.
    """
    if delayed_LN.ndim != 2:
        raise ValueError(
            f"delayed_LN must be 2-D [L, N], got shape {tuple(delayed_LN.shape)}"
        )
    L, N = delayed_LN.shape
    T = L - (N - 1)
    if T <= 0:
        raise ValueError(
            f"delayed_LN has L={L}, N={N}; need L >= N so at least one "
            f"data row can be recovered."
        )
    out = torch.empty((T, N), device=delayed_LN.device, dtype=delayed_LN.dtype)
    for c in range(N):
        out[:, c] = delayed_LN[c : c + T, c]
    return out


__all__ = ["BOC_ID", "EOC_ID", "apply_delay_pattern", "reverse_delay_pattern"]
