# Adapted from https://github.com/fla-org/flash-linear-attention/pull/677
 
import pytest
from torch.testing import assert_close
import torch
import torch.nn.functional as torch_functional
import torch_npu
 
from gdn_tri_inverse.core import (
    inv_tril_inplace,
    recurrent_gated_delta_rule_ref,
    chunk_gated_delta_rule_ref,
)
 
device = "npu:0"  # pick an available device
 
 
@pytest.mark.parametrize("chunk_size", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("batch", [1, 17])
@pytest.mark.parametrize(
    "dtype,atol,rtol", [(torch.float32, 1e-5, 1e-5), (torch.float16, 1e-2, 1e-2)]
)
@torch.inference_mode()
def test_inv_tril(chunk_size, batch, dtype, atol, rtol):
    torch.manual_seed(0)
    shape = (batch, chunk_size, chunk_size)
 
    # NOTE: scale-down off-diagonals to avoid bad condition number
    scale = 0.1 if chunk_size >= 64 else 0.2
    A = scale * torch.abs(torch.rand(*shape, device=device, dtype=dtype))
    A = torch.tril(A, diagonal=-1)
    A_inv = A.clone()
    inv_tril_inplace(A_inv)
 
    I = torch.eye(chunk_size, device=device, dtype=dtype)
    I_recover = (I - A) @ (I + A_inv)
    assert_close(I_recover, I.expand_as(I_recover), atol=atol, rtol=rtol)
 
 
@pytest.mark.parametrize(
    ("B", "T", "H", "D", "scale", "gate_logit_normalizer", "mask_p", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-{}".format(
                *test
            ),
        )
        for test in [
            (1, 63, 1, 64, 1, 1, 0, torch.float16),
            (2, 500, 3, 60, 1, 1, 0, torch.float16),
            (2, 1000, 3, 64, 0.1, 1, 0.5, torch.float16),
            (3, 1024, 4, 100, 1, 0.1, 0, torch.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, torch.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, torch.float16),
            (2, 1500, 4, 128, 0.1, 10, 0, torch.float16),
            (4, 2048, 8, 64, 0.1, 1, 0, torch.float16),
        ]
    ],
)
@torch.inference_mode()
def test_chunk_forward_ref(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    mask_p: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
 
    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g = torch_func.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    g = g / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = map(lambda x: x.to(device), (q, k, v, beta, g, h0))
 
    q = torch_func.normalize(q.clone(), p=2, dim=-1)
    k = torch_func.normalize(k.clone(), p=2, dim=-1)
 
    chunk_o, chunk_ht = chunk_gated_delta_rule_ref(
        q=torch_func.normalize(q.clone(), p=2, dim=-1),
        k=torch_func.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )
 
    ref_o, ref_ht = recurrent_gated_delta_rule_ref(
        q=torch_func.normalize(q.clone(), p=2, dim=-1),
        k=torch_func.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )
 
    assert_close(chunk_o, ref_o)
    assert_close(chunk_ht, ref_ht)