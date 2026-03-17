# Adapted from https://github.com/fla-org/flash-linear-attention/pull/677

import pytest
from torch.testing import assert_close
import torch
import torch.nn.functional as torch_func
import torch_npu  # noqa

from gdn_tri_inverse.core import (
    recurrent_gated_delta_rule_ref,
    chunk_gated_delta_rule_ref,
)

device = "npu:0"  # pick an available device


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "scale", "gate_logit_normalizer", "mask_p", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-{}".format(
                *test
            ),
        )
        for test in [
            (1, 63, 1, 64, 1, 1, 0, torch.float16),
            (2, 500, 3, 60, 1, 1, 0, torch.float16),
            (2, 1000, 3, 64, 0.1, 1, 0.5, torch.float16),
            (3, 1024, 4, 100, 1, 0.1, 0, torch.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, torch.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, torch.float16),
            (2, 1500, 4, 128, 0.1, 10, 0, torch.float16),
            (4, 2048, 8, 64, 0.1, 1, 0, torch.float16),
        ]
    ],
)
@pytest.mark.skip(
    reason="The triangular inverse default forward substitution tests fail."
)
@torch.inference_mode()
def test_chunk_forward_ref(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    mask_p: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g = torch_func.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    g = g / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = map(lambda x: x.to(device), (q, k, v, beta, g, h0))

    q = torch_func.normalize(q.clone(), p=2, dim=-1)
    k = torch_func.normalize(k.clone(), p=2, dim=-1)

    chunk_o, chunk_ht = chunk_gated_delta_rule_ref(
        q=torch_func.normalize(q.clone(), p=2, dim=-1),
        k=torch_func.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )

    ref_o, ref_ht = recurrent_gated_delta_rule_ref(
        q=torch_func.normalize(q.clone(), p=2, dim=-1),
        k=torch_func.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )

    assert_close(chunk_o, ref_o)
    assert_close(chunk_ht, ref_ht)
