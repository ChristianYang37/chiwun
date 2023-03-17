# __author__ == ChiWun Yang
# email == christiannyang37@gmail.com

import torch
import torch.nn as nn


class PairWiseLoss(nn.Module):
    def __init__(self):
        super(PairWiseLoss, self).__init__()
        self.sigmoid = nn.LogSigmoid()

    def forward(self, prediction):
        n = prediction.shape[0]
        cnt = 0
        loss = torch.Tensor([0]).to(prediction.device)
        for i in range(n):
            for j in range(i + 1, n):
                loss += self.sigmoid(prediction[i] - prediction[j])
                cnt += 1
        return -loss / cnt
