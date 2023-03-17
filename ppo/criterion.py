# __author__ == ChiWun Yang
# email == christiannyang37@gmail.com

import torch
import numpy as np
import torch.nn as nn


class ObjectiveLoss(nn.Module):
    """
        see https://arxiv.org/pdf/2204.05862v1.pdf
    """
    def __init__(self, init_beta, init_gamma, kl_target=6, k_beta=0.1):
        super(ObjectiveLoss, self).__init__()
        self.beta = init_beta
        self.gamma = init_gamma
        self.kl_target = kl_target
        self.k_beta = k_beta

    def forward(self, rl_logit, sft_logit, reward, ce_loss=0):
        kl = torch.abs(torch.log(rl_logit) - torch.log(sft_logit)).mean()
        beta = self.beta
        rl_max_values, rl_max_indices = rl_logit.max(2)
        sft_values = torch.cat(
            [sft_logit[i][:, rl_max_indices[i]].diag().unsqueeze(0) for i in range(rl_logit.shape[0])])
        advantage = torch.norm((torch.log(rl_max_values) - torch.log(sft_values)) * reward).mean()

        loss = beta * kl - advantage + self.gamma * ce_loss

        # update beta
        e = np.clip(abs(float(kl)) / self.kl_target - 1, -0.2, 0.2)
        self.beta *= 1 + self.k_beta * e

        return loss, kl, beta
