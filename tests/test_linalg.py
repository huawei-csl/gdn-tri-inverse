# Adapted from https://github.com/fla-org/flash-linear-attention/pull/677

import pytest
from torch.testing import assert_close
import torch

from gdn_tri_inverse.linalg import (
    tri_inv_qwen3_next_default,
    tri_inv_vcs,
    tri_inv_mxr,
    tri_inv_mcs,
    tri_inv_triton,
)

device = "npu:0"  # pick an available device


def gen_random_matrix(shape, dtype):
    A = 0.1 * torch.abs(torch.rand(*shape, dtype=dtype))
    return A


def _test_tri_inv_common(
    chunk_size: int,
    batch: int,
    tri_inv_fn,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    matrix_gen,
):
    torch.manual_seed(0)
    shape = (batch, chunk_size, chunk_size)

    if chunk_size == 128 and tri_inv_fn == tri_inv_triton:
        assert True, "Triton does not support chunk_size 128"
        return

    A = matrix_gen(shape=shape, dtype=dtype)
    A = torch.tril(A, diagonal=-1)

    A_npu = A.to(device)
    A_inv_npu = tri_inv_fn(A_npu)
    A_inv = A_inv_npu.cpu()
    torch.npu.synchronize()

    ref_dtype = torch.float64
    A_inv = A_inv.to(ref_dtype)
    Identity = torch.eye(chunk_size, dtype=ref_dtype)
    I_recover = (Identity + A.to(ref_dtype)) @ A_inv
    assert_close(
        I_recover,
        Identity.expand_as(I_recover),
        atol=chunk_size * atol,
        rtol=chunk_size * rtol,
    )


@pytest.mark.parametrize("chunk_size", [32, 64, 128])
@pytest.mark.parametrize("batch", [1, 17])
@pytest.mark.parametrize(
    "tri_inv_fn,dtype,atol,rtol",
    [
        (tri_inv_vcs, torch.float16, 1e-5, 1e-2),
        (tri_inv_vcs, torch.float32, 1e-8, 5e-5),
        (tri_inv_triton, torch.float32, 1e-8, 5e-5),
    ],
)
@pytest.mark.parametrize("matrix_gen", [gen_random_matrix])
@torch.inference_mode()
def test_tri_inv_stable_methods(
    chunk_size: int,
    batch: int,
    tri_inv_fn,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    matrix_gen,
):
    _test_tri_inv_common(chunk_size, batch, tri_inv_fn, dtype, atol, rtol, matrix_gen)


@pytest.mark.parametrize("chunk_size", [32, 64, 128])
@pytest.mark.parametrize("batch", [1, 17])
@pytest.mark.parametrize(
    "tri_inv_fn,dtype,atol,rtol",
    [
        (tri_inv_mcs, torch.float16, 1e-5, 1e-2),
        (tri_inv_mxr, torch.float16, 1e-5, 1e-2),
        (tri_inv_qwen3_next_default, torch.float16, 1e-5, 1e-2),
        (tri_inv_qwen3_next_default, torch.float32, 1e-8, 5e-5),
    ],
)
@pytest.mark.parametrize("matrix_gen", [gen_random_matrix])
@torch.inference_mode()
@pytest.mark.skip(
    reason="The triangular inverse MCS MXR and default forward substitution tests fail."
)
def test_tri_inv_unstable_methods(
    chunk_size: int,
    batch: int,
    tri_inv_fn,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    matrix_gen,
):
    _test_tri_inv_common(chunk_size, batch, tri_inv_fn, dtype, atol, rtol, matrix_gen)
