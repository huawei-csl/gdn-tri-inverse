#!/usr/bin/env python3
"""
Patches sgl_kernel_npu/fla/chunk.py to support the SOLVE_TRIL_BACKEND env var.

Run from the sgl-kernel-npu repository root before building the wheel:

    python /tmp/apply_backend_selector.py

Supported values for SOLVE_TRIL_BACKEND (set at container start time):
    default   -- keep the original solve_tril_npu triton kernel (no change)
    pto-vcs   -- use pto_kernels.pto_tri_inv (vector column sweep)
    pto-mxr   -- use pto_kernels.pto_tri_inv_rec_unroll (matmul recursive unroll)

The script is idempotent: it exits cleanly if chunk.py is already patched.
It exits with a non-zero status if the expected target string is not found,
so the Docker build fails fast when sgl_kernel_npu changes the call site.
"""

import re
import sys
from pathlib import Path

CHUNK_PY = Path("python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk.py")

# Code inserted before the first module-level function definition.
HEADER = '''\
import os as _os
import torch.nn.functional as _F


def _make_gdn_inv_fn():
    """
    Reads SOLVE_TRIL_BACKEND at import time and returns the matching inv function,
    or None to keep the default solve_tril_npu triton kernel.
    """
    _backend = _os.getenv("SOLVE_TRIL_BACKEND", "default")

    if _backend == "pto-mxr":
        import torch
        from pto_kernels import pto_tri_inv_rec_unroll as _kernel

        def _fn(A, cu_seqlens=None, output_dtype=None):
            print("Using pto-mxr backend.")
            if cu_seqlens is not None:
                A_inv = _kernel(A.to(torch.float16), cu_seqlens=cu_seqlens, is_bsnd_format=True, is_lower=True)
            else:
                A_inv = _kernel(A.to(torch.float16), is_bsnd_format=True, is_lower=True)
            return A_inv.to(output_dtype) if output_dtype is not None else A_inv

        return _fn

    return None  # "default" -> keep solve_tril


_GDN_INV_FN = _make_gdn_inv_fn()

'''

OLD_CALL = "    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)"

NEW_CALL = """\
    if _GDN_INV_FN is not None:
        A = _GDN_INV_FN(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    else:
        A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)"""

IDEMPOTENCY_MARKER = "_GDN_INV_FN"


def patch(path: Path) -> None:
    if not path.exists():
        print(
            f"ERROR: {path} not found -- run from the sgl-kernel-npu repository root",
            file=sys.stderr,
        )
        sys.exit(1)

    text = path.read_text(encoding="utf-8")

    if IDEMPOTENCY_MARKER in text:
        print(f"SKIP: {path} is already patched")
        return

    if OLD_CALL not in text:
        print(
            f"ERROR: target string not found in {path}\n"
            f"Expected: {OLD_CALL!r}\n"
            "The sgl_kernel_npu version may have changed -- review the patch.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Insert HEADER immediately before the first module-level 'def'.
    match = re.search(r"^def ", text, re.MULTILINE)
    if match is None:
        print(f"ERROR: no module-level 'def' found in {path}", file=sys.stderr)
        sys.exit(1)

    text = text[: match.start()] + HEADER + text[match.start() :]
    text = text.replace(OLD_CALL, NEW_CALL, 1)

    path.write_text(text, encoding="utf-8")
    print(f"Patched {path}")


if __name__ == "__main__":
    patch(CHUNK_PY)
