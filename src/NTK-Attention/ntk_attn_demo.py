import torch
import torch.nn as nn


def phi(x):
    return nn.ELU()(x / (x.size(-1) ** 0.25)) + 1


def ntk_attn(query_layer, key_layer, value_layer, sum_phi_k, sum_phi_kv, causal_mask=True):
    """

    :param query_layer: shape[bs, h, nq, d]
    :param key_layer: shape[bs, h, nk, d]
    :param value_layer: shape[bs, h, nk, d]
    :param sum_phi_k: shape[1, h, d, 1]
    :param sum_phi_kv: shape[1, h, d, d]
    :param causal_mask: bool, default to True
    :return: context_layer
    """
    dtype = query_layer.dtype

    query_layer, key_layer, value_layer = query_layer.float(), key_layer.float(), value_layer.float()
    k = sum_phi_k.abs().float()
    Z = sum_phi_kv.float()

    bs, h = query_layer.size()[:2]
    nq, nk = query_layer.size(-2), key_layer.size(-2)
    d = query_layer.size(-1)

    if causal_mask:
        mask = torch.tril(torch.ones(bs, h, nq, nk), diagonal=nk - nq).to(query_layer.device)
    else:
        mask = None

    phi_q = phi(query_layer)

    A = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / (d ** 0.5)
    max_A = A.max(-1).values.unsqueeze(-1)
    exp_max_A = torch.exp(max_A)
    A = torch.exp(A - max_A)

    if mask is not None:
        A *= mask

    D = A.sum(-1).unsqueeze(-1) + torch.matmul(phi_q, k) / exp_max_A
    context_layer = (torch.matmul(A, value_layer) + torch.matmul(phi_q, Z) / exp_max_A) / D

    return context_layer.contiguous().to(dtype)
