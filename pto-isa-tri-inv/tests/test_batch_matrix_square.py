import torch
import pto_isa_kernels
import pytest
import numpy as np
import os
import random

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)


@pytest.mark.parametrize("block_dim", [1, 2, 3, 5, 8, 11, 16, 37, 64, 128, 256])
@pytest.mark.parametrize("matrix_size", [16, 32, 64, 96, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)
def test_pto_isa_batch_matrix_square(
    block_dim: int, matrix_size: int, dtype: torch.dtype
):
    x = torch.rand((block_dim, matrix_size, matrix_size), device="cpu", dtype=dtype)
    x_npu = x.npu()
    torch.npu.synchronize()
    z_npu = torch.ops.npu.batch_matrix_square(x_npu)
    torch.npu.synchronize()
    z = z_npu.cpu()
    ref = (x.to(torch.double) @ x.to(torch.double)).to(torch.float32)
    assert torch.allclose(z, ref)
