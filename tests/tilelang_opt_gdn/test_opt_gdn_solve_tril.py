import pytest
import tilelang
import torch
import torch.nn.functional as F

from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_solve_tril import (
    solve_tril,
    ref_solve_tril,
    ref_chunk_cumsum,
    ref_kkt,
)


@pytest.mark.parametrize(
    ("B", "H", "L", "DK", "C"),
    [
        pytest.param(1, 1, 128, 128, 128, id="B1-H1-L128-DK128-C128"),
    ],
)
def test_solve_tril(B, H, L, DK, C):
    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    beta = torch.rand((B, H, L)).npu().to(torch.float16)
    g = torch.randn((B, H, L)).npu().to(torch.float)
    k = F.normalize(k, dim=-1, p=2)
    g = F.logsigmoid(g)
    g = ref_chunk_cumsum(g, C)
    a = ref_kkt(k, beta, g, C)

    o = solve_tril(a)
    ref_o = ref_solve_tril(a)

    torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=1e-3, atol=1e-3)
