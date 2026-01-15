# Adapted from https://github.com/fla-org/flash-linear-attention/pull/677
 
import pytest
from torch.testing import assert_close
import torch
import torch.nn.functional as torch_func
import torch_npu
 
from gdn_tri_inverse.linalg import (
    inv_tril_inplace, tri_inv_vcs
)
 
device = "npu:0"  # pick an available device

def gen_random_matrix(shape, dtype):
    A = 0.1 * torch.abs(torch.rand(*shape, dtype=dtype))
    A = torch.tril(A, diagonal=-1)
    return A

 
@pytest.mark.parametrize("chunk_size", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("batch", [1, 17])
@pytest.mark.parametrize(
    "dtype,atol,rtol", [(torch.float32, 1e-8, 5e-5), (torch.float16, 1e-5, 1e-2)]
)
@pytest.mark.parametrize('matrix_gen', [gen_random_matrix])
@torch.inference_mode()
def test_inv_tril_inplace(chunk_size, batch, dtype, atol, rtol, matrix_gen):
    torch.manual_seed(0)
    shape = (batch, chunk_size, chunk_size)
 
    A = matrix_gen(shape=shape, dtype=dtype)

    A_npu = A.to(device)
    torch.npu.synchronize()
    A_inv_npu = A.clone()
    torch.npu.synchronize()
    inv_tril_inplace(A_inv_npu)
    torch.npu.synchronize()
    A_inv = A_inv_npu.cpu()
    torch.npu.synchronize()
 
    ref_dtype = torch.float64
    A_inv = A_inv.to(ref_dtype)
    I = torch.eye(chunk_size, dtype=ref_dtype)
    I_recover = (I - A.to(ref_dtype)) @ (I + A_inv)
    assert_close(I_recover, I.expand_as(I_recover), atol=chunk_size*atol, rtol=chunk_size*rtol)


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@pytest.mark.parametrize("batch", [1, 17])
@pytest.mark.parametrize(
    "dtype,atol,rtol", [(torch.float32, 1e-8, 5e-5), (torch.float16, 1e-5, 1e-2)]
)
@pytest.mark.parametrize('matrix_gen', [gen_random_matrix])
@torch.inference_mode()
def test_tri_inv_vcs(chunk_size, batch, dtype, atol, rtol, matrix_gen):
    torch.manual_seed(0)
    shape = (batch, chunk_size, chunk_size)
 
    A = matrix_gen(shape=shape, dtype=dtype)

    A_npu = A.to(device)
    torch.npu.synchronize()
    A_inv_npu = tri_inv_vcs(A_npu)
    torch.npu.synchronize()
    A_inv = A_inv_npu.cpu()
    torch.npu.synchronize()
 
    ref_dtype = torch.float64
    A_inv = A_inv.to(ref_dtype)
    I = torch.eye(chunk_size, dtype=ref_dtype)
    I_recover = (I + A.to(ref_dtype)) @ A_inv
    assert_close(I_recover, I.expand_as(I_recover), atol=chunk_size * atol, rtol=chunk_size * rtol)
 