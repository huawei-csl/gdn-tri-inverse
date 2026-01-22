import torch
from tcuscan import run_tri_inv_col_sweep, run_triu_inv_rec_unroll  # noqa


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
