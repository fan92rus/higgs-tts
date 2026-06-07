"""Print the correct PyTorch wheel index URL for the installed NVIDIA GPU.

Used by the installer to pick the right torch build by compute capability:
  - Blackwell (RTX 50xx, B100/B200; cc >= 10.0)  -> cu128  (torch >= 2.7, has sm_120)
  - Pascal..Hopper (GTX 10xx..H100; cc 5.0-9.x)  -> cu121  (torch 2.5.1)
  - unknown / no nvidia-smi                       -> cu121  (safe default)

Uses only the standard library + nvidia-smi (runs BEFORE torch is installed).
Prints exactly one line (the index URL) to stdout.
"""

from __future__ import annotations

import re
import subprocess

CU121 = "https://download.pytorch.org/whl/cu121"
CU128 = "https://download.pytorch.org/whl/cu128"


def _query(field: str) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def _max_compute_cap() -> float | None:
    caps: list[float] = []
    for v in _query("compute_cap"):
        try:
            caps.append(float(v))
        except ValueError:
            pass
    return max(caps) if caps else None


def _looks_blackwell_by_name() -> bool:
    # Consumer Blackwell: RTX 5050/5060/5070/5080/5090 (+ Ti). Fallback when
    # the driver doesn't report compute_cap.
    names = " ".join(_query("name"))
    return bool(re.search(r"RTX\s*50\d0", names, re.IGNORECASE))


def main() -> None:
    cc = _max_compute_cap()
    # Blackwell (sm_100/sm_120 == cc 10.0/12.0) needs cu128; older cards use cu121.
    if (cc is not None and cc >= 10.0) or (cc is None and _looks_blackwell_by_name()):
        print(CU128)
    else:
        print(CU121)


if __name__ == "__main__":
    main()
