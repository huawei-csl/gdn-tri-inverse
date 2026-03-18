import pytest
import tilelang
import torch

from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_chunk_scaled_dot_kkt import kkt, ref_kkt


@pytest.mark.parametrize(
    ("B", "H", "L", "DK", "C"),
    [
        pytest.param(2, 16, 16384, 128, 128, id="B2-H16-L16384-DK128-C128"),
    ],
)
def test_kkt(B, H, L, DK, C):
    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    beta = torch.rand((B, H, L)).npu().to(torch.float16)
    g = torch.randn((B, H, L)).npu().to(torch.float)

    a = kkt(k, beta, g, C)
    ref_a = ref_kkt(k, beta, g, C)

    torch.testing.assert_close(a.cpu(), ref_a.cpu(), rtol=1e-3, atol=1e-3)
