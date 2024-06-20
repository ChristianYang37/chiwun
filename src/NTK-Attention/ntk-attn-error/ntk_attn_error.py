import torch
import torch.nn as nn
from torch.optim import Adam
import os
from tqdm import tqdm


def set_seed(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


set_seed()


repeat_times = 20
d = 64
L = 128
train_size = 40000
val_size = 8000

m_list = [2 ** (i+1) for i in range(18)]


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


def main():
    result = []
    W_Q, W_K, W_V = torch.randn(d, d), torch.randn(d, d), torch.randn(d, d)
    for m in m_list:
        # B = math.log(m)
        P = torch.randn(m, d)
        Z, k = torch.randn(d, d), torch.randn(d)
        Z.requires_grad = True
        k.requires_grad = True
        optim = Adam([Z, k], lr=1e-3)
        data_iter = tqdm(range(train_size))
        for _ in data_iter:
            X = torch.randn(L, d)
            X = torch.cat([P, X])
            Q, K, V = X @ W_Q, X @ W_K, X @ W_V
            Y = torch.softmax(Q @ K.T / (d ** 0.5), dim=-1) @ V
            Y_hat = ntk_attn(Q.reshape(1, 1, -1, d), K.reshape(1, 1, -1, d), V.reshape(1, 1, -1, d), k, Z,
                             causal_mask=False).reshape(-1, d)
            loss = (Y - Y_hat) ** 2

            optim.zero_grad()

            loss.backward()

            optim.step()

            data_iter.set_postfix(loss=loss.item())

        error_sum = 0
        data_iter = tqdm(range(val_size))
        for _ in data_iter:
            X = torch.randn(L, d)
            X = torch.cat([P, X])
            Q, K, V = X @ W_Q, X @ W_K, X @ W_V
            QK = Q @ K.T
            Y = torch.softmax(QK / (d ** 0.5), dim=-1) @ V
            Y_hat = ntk_attn(Q.reshape(1, 1, -1, d), K.reshape(1, 1, -1, d), V.reshape(1, 1, -1, d), k, Z,
                             causal_mask=False).reshape(-1, d)

            error = ((Y - Y_hat) ** 2).sum().item()
            if QK > 1:
                error /= QK.max()

            error_sum += error

        result.append(error_sum / (L * d * val_size))

    print(result)
    with open("./result.txt", encoding="utf-8", mode="w") as file:
        file.write(str(result))


main()
