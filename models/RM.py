# __author__ == ChiWun Yang
# email == christiannyang37@gmail.com

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from config import SFT_huggingface_card


class RM(nn.Module):
    def __init__(self, model_path='./rm_model'):
        super(RM, self).__init__()
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(SFT_huggingface_card)
        self.pad_id = self.tokenizer.pad_token_id

        if os.path.exists(model_path):
            self.transformer = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            self.transformer = AutoModelForSeq2SeqLM.from_pretrained(SFT_huggingface_card)
            self.transformer.save_pretrained(model_path)

    def forward(self, input_ids, decoder_input_ids):
        batch_size, seq_len = input_ids.shape[0], decoder_input_ids.shape[1]
        x = torch.zeros(size=(batch_size, 1)).to(input_ids.device)
        output = []
        for i in range(seq_len - 1):
            logit = self.transformer(input_ids, decoder_input_ids=decoder_input_ids[:, :i + 1]).logits[:, -1]
            log_logit = F.log_softmax(logit, dim=1)
            x += log_logit[:, decoder_input_ids[:, i + 1]].diag().reshape(-1, 1)
            output.append(logit[:, decoder_input_ids[:, i + 1]].diag().reshape(-1, 1))
        return torch.cat(output, dim=1), x

    def scoring(self, input_ids, logit, device):
        """Calculate Reward"""
        self.eval()
        self.to(device)
        decoder_input_ids = logit.argmax(2)
        pad_ids = torch.LongTensor([[self.pad_id] for _ in range(input_ids.shape[0])]).to(device)
        decoder_input_ids = torch.cat([pad_ids, decoder_input_ids], dim=1)
        rewards = self.forward(input_ids, decoder_input_ids)[0]

        return rewards

    def save(self):
        self.transformer.save_pretrained(self.model_path)
