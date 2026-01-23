import torch
import op_extension
import pytest


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)
def test_add_custom_ops(dtype: torch.dtype):
    # Define the tensor size
    length = [20, 2048]
    # Create random input tensors on CPU with float16 data type
    x = torch.rand(length, device="cpu", dtype=dtype)
    y = torch.rand(length, device="cpu", dtype=dtype)

    x_npu = x.npu()
    y_npu = y.npu()
    # Call the custom my_add operator
    output = torch.ops.npu.my_add(x_npu, y_npu).cpu()
    # Compute the expected result using standard addition
    cpuout = torch.add(x, y)

    # Validate the results
    assert torch.allclose(output, cpuout)
