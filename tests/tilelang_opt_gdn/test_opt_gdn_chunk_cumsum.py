import pytest
import tilelang
import torch

from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_chunk_cumsum import (
    chunk_cumsum,
    ref_chunk_cumsum,
)


@pytest.mark.parametrize(
    ("B", "H", "L", "C"),
    [
        pytest.param(2, 16, 16384, 128, id="B2-H16-L16384-C128"),
    ],
)
def test_chunk_cumsum(B, H, L, C):
    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    g = torch.randn((B, H, L)).npu().to(torch.float)
    g_sum = chunk_cumsum(g, C)
    ref_g_sum = ref_chunk_cumsum(g, C)
    torch.testing.assert_close(g_sum.cpu(), ref_g_sum.cpu(), rtol=1e-5, atol=1e-5)
