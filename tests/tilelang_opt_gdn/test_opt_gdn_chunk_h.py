import pytest
import tilelang
import torch
import torch.nn.functional as F

from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_chunk_cumsum import ref_chunk_cumsum
from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_chunk_h import chunk_h, ref_chunk_h


@pytest.mark.parametrize(
    ("B", "H", "L", "DK", "DV", "C"),
    [
        pytest.param(2, 16, 16384, 128, 128, 128, id="B2-H16-L16384-DK128-DV128-C128"),
    ],
)
def test_chunk_h(B, H, L, DK, DV, C):
    tilelang.cache.clear_cache()
    torch.manual_seed(0)
    torch.set_printoptions(threshold=float("inf"), sci_mode=True)

    k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    w = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    u = torch.randn((B, H, L, DV)).npu().to(torch.float16)
    g = torch.randn((B, H, L)).npu().to(torch.float)
    g = F.logsigmoid(g)
    k, w = F.normalize(k, dim=-1, p=2), F.normalize(w, dim=-1, p=2)
    g = ref_chunk_cumsum(g, C)

    s, new_v, final_s = chunk_h(k, w, u, g, C)
    ref_s, ref_new_v, ref_final_s = ref_chunk_h(k, w, u, g, C)

    torch.testing.assert_close(s.cpu(), ref_s.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(new_v.cpu(), ref_new_v.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(final_s.cpu(), ref_final_s.cpu(), rtol=1e-5, atol=1e-5)
