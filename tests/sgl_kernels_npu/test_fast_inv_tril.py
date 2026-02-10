"""
Docstring for tests.sgl_kernels_npu.test_fast_inv_tril

This unit test demonstrates the replacement of the `inv_tril_inplace` with the `fast_inv_tril` (which is based on `torch.ops.npu.triangular_inverse`) on sgl-kernel-npu repository.

Code is copied from https://github.com/sgl-project/sgl-kernel-npu/blob/2026.02.01/tests/python/sgl_kernel_npu/test_solve_tril.py

Important: both methods `fast_inv_tril` and `inv_tril_inplace` compute the inverse of (I-A)!
"""

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel_npu.fla.chunk import fast_inv_tril, inv_tril_inplace

NPU_DEVICE = "npu:1"
torch.npu.set_device(1)


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    else:
        assert error_rate < ratio, msg


def print_diff(name, ref, tri, atol=0.005):
    abs_diff = torch.abs(ref - tri)
    max_abs_diff = abs_diff.max().item()
    print(f"[{name}] Max absolute difference: {max_abs_diff:.6f}")
    if max_abs_diff > atol:
        print(f"Exceeds tolerance ({atol})!")


@pytest.mark.parametrize(
    ("B", "T", "H", "chunk_size"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-chunk_size{}".format(*test))
        for test in [
            (1, 63, 1, 16),
            (2, 500, 4, 32),
            (2, 1000, 5, 64),
            (3, 1024, 6, 64),
            (4, 2048, 8, 64),
        ]
    ],
)
@pytest.mark.parametrize("tri_inv_fnc", [fast_inv_tril, inv_tril_inplace])
def test_solve_tril(B: int, T: int, H: int, chunk_size: int, tri_inv_fnc):
    # do not randomly initialize A otherwise the inverse is not stable
    k = F.normalize(
        torch.randn((B, H, T, 64), dtype=torch.float32, device=NPU_DEVICE), dim=-1
    )
    # Pad the second-to-last dimension (T) to be a multiple of chunk_size
    padding_size = (chunk_size - T % chunk_size) % chunk_size
    k_padded = F.pad(k, (0, 0, 0, padding_size, 0, 0, 0, 0))
    k_padded = k_padded.reshape(B, H, -1, chunk_size, 64)
    A = (k_padded @ k_padded.transpose(-1, -2)).tril(-1)

    ref = torch.inverse(
        torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)[None, None, None, ...]
        - A
    )
    torch.npu.synchronize()
    ref = ref.reshape(B, H, -1, chunk_size)[:, :, :T, :]

    torch.npu.synchronize()
    tri = tri_inv_fnc(A)
    tri = tri.reshape(B, H, -1, chunk_size)[:, :, :T, :]
    torch.npu.synchronize()

    assert_close("solve_tril", ref, tri, 0.0001)
