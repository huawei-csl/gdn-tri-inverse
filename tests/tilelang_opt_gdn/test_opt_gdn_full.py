import pytest
import tilelang
import torch
import torch.nn.functional as F

from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_chunk_cumsum import cumsum_ker
from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_chunk_h import chunk_h_ker
from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_chunk_o import chunk_o_ker
from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_chunk_scaled_dot_kkt import kkt_ker
from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_solve_tril import (
    solve_tril_ker,
    solve_tril_64_ker,
    solve_tril_128_ker,
)
from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_wy_fast import wy_fast_ker
from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_full import ref_seq_gdn


@pytest.mark.parametrize(
    ("B", "H", "L", "DK", "DV", "C", "BK", "BV"),
    [
        pytest.param(
            1,
            2,
            1024,
            128,
            128,
            128,
            128,
            128,
            id="B1-H2-L1024-DK128-DV128-C128-BK128-BV128",
        ),
    ],
)
def test_full_gdn(B, H, L, DK, DV, C, BK, BV):
    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    q = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    v = torch.randn((B, H, L, DV)).npu().to(torch.float16)
    q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
    g = torch.randn((B, H, L)).npu().to(torch.float)
    g = F.logsigmoid(g)
    beta = torch.rand((B, H, L)).npu().to(torch.float16)

    ker1 = cumsum_ker(B, H, L, C)
    ker2 = kkt_ker(B, H, L, DK, C, BK)
    if C == 32:
        ker3 = solve_tril_ker(B, H, L, C)
    elif C == 64:
        ker3 = solve_tril_64_ker(B, H, L)
    elif C == 128:
        ker3 = solve_tril_128_ker(B, H, L)
    ker4 = wy_fast_ker(B, H, L, DK, DV, C, BK, BV)
    ker5 = chunk_h_ker(B, H, L, DK, DV, C, BK, BV)
    ker6 = chunk_o_ker(B, H, L, DK, DV, C, BK, BV)

    idt = torch.eye(C).npu().to(torch.float)
    msk1 = torch.tril(torch.ones((C, C)), diagonal=-1).npu().to(torch.float)
    msk2 = torch.tril(torch.ones((C, C)), diagonal=0).npu().to(torch.float)
    workspace = (
        torch.zeros((B * H * ((DV + BV - 1) // BV), DK, BV)).npu().to(torch.float16)
    )
    s = torch.zeros((B, H, (L + C - 1) // C, DK, DV)).npu().to(torch.float16)

    g_sum = ker1(g)
    a = ker2(k, beta, g_sum, msk1)
    a = ker3(a, idt)
    w, u = ker4(k, v, beta, g_sum, a)
    nv, fs = ker5(k, w, u, g_sum, workspace, s)
    o = ker6(q, k, nv, s, g_sum, msk2)
    ref_o = ref_seq_gdn(q, k, v, g, beta)

    torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=1e-3, atol=1e-3)
