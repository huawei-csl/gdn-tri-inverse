import torch
from typing import Optional
from tcuscan import (
    run_tri_inv_col_sweep,
    run_triu_inv_rec_unroll,
    run_tri_inv_cube_col_sweep,
)
from sgl_kernel_npu.fla.solve_tril import solve_tril_npu


def inv_tril_inplace(A: torch.Tensor):
    """
    compute inv(I - A) where A is strict lower-triangular. Adapted from:
    https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L485-L490
    The algorithm is somewhat "in-place", in the sense that it does not
    explicitly form a full matrix for the inverse.
    """
    assert A.shape[-2] == A.shape[-1]
    chunk_size = A.shape[-1]
    for i in range(1, chunk_size):
        row = A[..., i, :i].clone()
        sub = A[..., :i, :i].clone()
        A[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)


def tri_inv_qwen3_next_default(A: torch.Tensor):
    """
    This is a wrapper to the inv_tril_inplace defined above.
    It is the default kernel used by qwen3 next.
    We only define this function in order to have a uniform
    signature for all inversion kernels.
    """
    A_inv = A.clone()
    inv_tril_inplace(A_inv)
    A_inv = A_inv + torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    return A_inv


def tri_inv_vcs(A: torch.Tensor) -> torch.Tensor:
    """
    VCS stands for Vector Column Sweep. The algorithm uses internally
    only vector units (AIV).
    """
    n = A.shape[-1]
    A_view = A.view(-1, n, n)
    A_inv = run_tri_inv_col_sweep(A_view)
    return A_inv.reshape(A.shape)


def tri_inv_mcs(A: torch.Tensor) -> torch.Tensor:
    """
    MCS stands for Matrix Column Sweep. The algorithm uses internally
    vector (AIV) and cube units (AIC).
    """
    n = A.shape[-1]
    A_view = A.view(-1, n, n)
    A_inv = run_tri_inv_cube_col_sweep(A_view)
    return A_inv.reshape(A.shape)


def tri_inv_mxr(A: torch.Tensor) -> torch.Tensor:
    """
    The individual matrices of the tensor A must be strictly
    upper triangular. The algorithm implements a matmul-based mixed
    recursive strategy (hence the MXR acronym) to compute the inverses.
    """
    n = A.shape[-1]
    A_view = A.view(-1, n, n)
    A_inv = run_triu_inv_rec_unroll(A_view)
    return A_inv.reshape(A.shape)


def tri_inv_triton(A: torch.Tensor):
    """
    Wrapper to the triton version of the triangular inverse from:
    https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/sgl_kernel_npu/fla/solve_tril.py

    The original triton version is adapted to the GDN layout and it expects the following format:
        [B, T, H, BT]
    Where:
        B and H are batch dimensions (respectively batch size and number of attention heads)
        BT is the chunk size, representing the size of the matrix
        T is the sequence length that is sliced into chunks of size BT (already padded)

    The wrapper taskes as an input a tensor of shape [*, BT, BT]
    """
    BT = A.shape[-1]
    B = 1 if A.dim() == 2 else A.numel() // (BT * BT)
    A_view = A.view(-1, BT, BT)
    A_view = A_view.reshape(1, 1, B * BT, BT).transpose(1, 2).contiguous()
    A_inv = solve_tril_npu(A_view)
    return A_inv.transpose(1, 2).reshape(B, BT, BT)
