import pytest
import tilelang
import torch

from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_wy_fast import wy_fast, ref_wy_fast


@pytest.mark.parametrize(
    ("B", "H", "L", "DK", "DV", "C"),
    [
        pytest.param(2, 16, 16384, 128, 128, 128, id="B2-H16-L16384-DK128-DV128-C128"),
    ],
)
def test_wy_fast(B, H, L, DK, DV, C):
    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    v = torch.randn((B, H, L, DV)).npu().to(torch.float16)
    beta = torch.rand((B, H, L)).npu().to(torch.float16)
    g = torch.randn((B, H, L)).npu().to(torch.float)
    a = torch.randn((B, H, L, C)).npu().to(torch.float16)

    w, u = wy_fast(k, v, beta, g, a, C)
    ref_w, ref_u = ref_wy_fast(k, v, beta, g, a, C)

    torch.testing.assert_close(w.cpu(), ref_w.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(u.cpu(), ref_u.cpu(), rtol=1e-5, atol=1e-5)
