import pytest
import tilelang
import torch
import torch.nn.functional as F

from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_chunk_o import chunk_o, ref_chunk_o


@pytest.mark.parametrize(
    ("B", "H", "L", "DK", "DV", "C"),
    [
        pytest.param(2, 16, 16384, 128, 128, 128, id="B2-H16-L16384-DK128-DV128-C128"),
    ],
)
def test_chunk_o(B, H, L, DK, DV, C):
    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    q = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    v = torch.randn((B, H, L, DV)).npu().to(torch.float16)
    s = torch.randn((B, H, (L + C - 1) // C, DK, DV)).npu().to(torch.float16)
    g = torch.randn((B, H, L)).npu().to(torch.float)
    q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)

    o = chunk_o(q, k, v, s, g, C)
    ref_o = ref_chunk_o(q, k, v, s, g, C)

    torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=1e-5, atol=1e-5)
