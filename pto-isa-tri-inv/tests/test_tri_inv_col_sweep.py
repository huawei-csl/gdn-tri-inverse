import torch
import op_extension
import pytest


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)
def test_add_custom_ops(dtype: torch.dtype):
    # Define the tensor size
    matrix_size = 64
    tile_len = matrix_size * matrix_size
    length = [20, tile_len]
    # Create random input tensors on CPU with float16 data type
    x = torch.rand(length, device="cpu", dtype=dtype)

    x_npu = x.npu()
    # Call the custom my_add operator
    output = torch.ops.npu.tri_inv(x_npu).cpu()
    # Compute the expected result using standard addition
    cpuout = torch.abs(x)

    # Validate the results
    assert torch.allclose(output, cpuout)
