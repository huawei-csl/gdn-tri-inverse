# Adapted from https://github.com/fla-org/flash-linear-attention/pull/677
 
import pytest
from torch.testing import assert_close
import torch
import torch.nn.functional as torch_func
import torch_npu
 
from gdn_tri_inverse.linalg import (
    tri_inv_qwen3_next_default, tri_inv_vcs, tri_inv_mxr
)
 
device = "npu:0"  # pick an available device

def gen_random_matrix(shape, dtype):
    A = 0.1 * torch.abs(torch.rand(*shape, dtype=dtype))
    return A

@pytest.mark.parametrize("chunk_size", [32, 64, 128])
@pytest.mark.parametrize("batch", [1, 17])
@pytest.mark.parametrize(
    "tri_inv_fn,dtype,atol,rtol", [
    (tri_inv_mxr, torch.float16, 1e-5, 1e-2),
    (tri_inv_vcs, torch.float16, 1e-5, 1e-2),
    (tri_inv_vcs, torch.float32, 1e-8, 5e-5),
    (tri_inv_qwen3_next_default, torch.float16, 1e-5, 1e-2),
    (tri_inv_qwen3_next_default, torch.float32, 1e-8, 5e-5)
    ]
)
@pytest.mark.parametrize('matrix_gen', [gen_random_matrix])
@torch.inference_mode()
def test_tri_inv(chunk_size, batch, tri_inv_fn, dtype, atol, rtol, matrix_gen):
    torch.manual_seed(0)
    shape = (batch, chunk_size, chunk_size)
 
    A = matrix_gen(shape=shape, dtype=dtype)
    A = torch.tril(A, diagonal=-1)

    A_npu = A.to(device)
    torch.npu.synchronize()
    A_inv_npu = tri_inv_fn(A_npu)
    torch.npu.synchronize()
    A_inv = A_inv_npu.cpu()
    torch.npu.synchronize()
 
    ref_dtype = torch.float64
    A_inv = A_inv.to(ref_dtype)
    I = torch.eye(chunk_size, dtype=ref_dtype)
    I_recover = (I + A.to(ref_dtype)) @ A_inv
    assert_close(I_recover, I.expand_as(I_recover), atol=chunk_size * atol, rtol=chunk_size * rtol)
