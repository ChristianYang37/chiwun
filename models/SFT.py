# __author__ == ChiWun Yang
# email == christiannyang37@gmail.com

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from config import SFT_huggingface_card


class SFT(nn.Module):
    def __init__(self):
        super(SFT, self).__init__()
        self.transformer = AutoModelForSeq2SeqLM.from_pretrained(SFT_huggingface_card)
        self.tokenizer = AutoTokenizer.from_pretrained(SFT_huggingface_card)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, **batch):
        output = self.transformer(**batch)
        output['logits'] = self.softmax(output['logits'])
        return output

    def generate(self, src_texts, use_cuda=False, do_sampling=False, num_sample=1):
        self.eval()
        device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.to(device)

        batch = self.tokenizer.prepare_seq2seq_batch(src_texts=src_texts)

        batch["input_ids"] = torch.LongTensor(batch["input_ids"]).to(device)
        batch["attention_mask"] = torch.LongTensor(batch["attention_mask"]).to(device)

        if do_sampling:
            prediction = self.transformer.generate(**batch, num_beams=num_sample, do_sampling=True,
                                                   num_return_sequences=num_sample)
            self.to("cpu")
            return self.tokenizer.batch_decode(prediction, skip_special_tokens=True)

        prediction = self.transformer.generate(**batch)
        return self.tokenizer.batch_decode(prediction, skip_special_tokens=True)
