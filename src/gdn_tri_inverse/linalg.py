import torch
from typing import Optional
from tcuscan import run_tri_inv_col_sweep

def inv_tril_inplace(A: torch.Tensor):
    """compute inv(I - A) where A is strict lower-triangular"""
    # Adapted from https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L485-L490
    assert A.shape[-2] == A.shape[-1]
    chunk_size = A.shape[-1]
    for i in range(1, chunk_size):
        row = A[..., i, :i].clone()
        sub = A[..., :i, :i].clone()
        A[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

def tri_inv_vcs(A: torch.Tensor) -> torch.Tensor:
    A_view = A
    if len(A.shape) == 4:
        A_view = A.view(-1, *A.shape[2:])
    A_inv = run_tri_inv_col_sweep(A_view)
    return A_inv.reshape(A.shape)


