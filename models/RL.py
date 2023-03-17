# __author__ == ChiWun Yang
# email == christiannyang37@gmail.com

import os
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from config import SFT_huggingface_card


class RL(nn.Module):
    def __init__(self, model_path):
        super(RL, self).__init__()
        self.model_path = model_path

        if os.path.exists(model_path):
            self.transformer = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            print(f"Download Model {SFT_huggingface_card}")
            self.transformer = AutoModelForSeq2SeqLM.from_pretrained(SFT_huggingface_card)
            self.tokenizer = AutoTokenizer.from_pretrained(SFT_huggingface_card)
            self.transformer.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, **batch):
        output = self.transformer(**batch)
        output['logits'] = self.softmax(output['logits'])
        return output

    def generate(self, src_texts, device):
        self.to(device)

        batch = self.tokenizer.prepare_seq2seq_batch(src_texts=src_texts)

        batch["input_ids"] = torch.LongTensor(batch["input_ids"]).to(device)
        batch["attention_mask"] = torch.LongTensor(batch["attention_mask"]).to(device)

        prediction = self.transformer.generate(**batch)
        return self.tokenizer.batch_decode(prediction, skip_special_tokens=True)

    def save(self):
        self.transformer.save_pretrained(self.model_path)