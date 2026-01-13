import torch


def ref_seq_gdn(q, k, v, g, beta):
    g = torch.exp(g)
    q = q.float()
    k = k.float()
    v = v.float()
    beta = beta.float()
    Batch, H, L, DK = q.shape
    DV = v.shape[-1]
    S = torch.zeros((Batch, H, DV, DK)).npu().to(torch.float)
    o = torch.empty((Batch, H, L, DV)).npu().to(torch.float)
    I = torch.eye(DK).npu().to(torch.float).view(1, 1, DK, DK)
    for i in range(0, L):
        q_i = q[:, :, i, :]
        k_i = k[:, :, i, :]
        v_i = v[:, :, i, :]
        beta_i = beta[:, :, i].view(Batch, H, 1, 1)
        g_i = g[:, :, i].view(Batch, H, 1, 1)
        kkt = k_i.unsqueeze(-1) * k_i.unsqueeze(-2)
        vkt = v_i.unsqueeze(-1) * k_i.unsqueeze(-2)
        A_i = g_i * (I - beta_i * kkt)
        term_1 = torch.matmul(S, A_i)
        term_2 = beta_i * vkt
        S = term_1 + term_2
        o[:, :, i, :] = torch.einsum("bhpq,bhq->bhp", S, q_i)
    return o.to(torch.float16)
