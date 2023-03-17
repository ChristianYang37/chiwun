# __author__ == ChiWun Yang
# email == christiannyang37@gmail.com

import torch
from tqdm import tqdm
from .criterion import PairWiseLoss
from .utils import score
from transformers import BertTokenizerFast


class rm_trainer:
    def __init__(self, rm, sft):
        self.rm = rm
        self.sft = sft
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

        self.loss_record = score(
            path='./outputs/rm_train_loss.png',
            xlabel='epochs',
            ylabel='rm_train_loss'
        )

    def train(self, dataloader, lr=1e-5):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device {device} to train")
        rm = self.rm.to(device)
        rm.train()
        optimizer = torch.optim.Adam(rm.parameters(), lr=lr)
        loss_fn = PairWiseLoss()

        loss_sum = 0
        data_iter = tqdm(dataloader)
        for input_texts in data_iter:
            src_text = input_texts[0]
            input_texts = input_texts[1:]
            batch = self.tokenizer.prepare_seq2seq_batch(src_texts=[src_text for _ in range(len(input_texts))],
                                                         tgt_texts=input_texts)
            input_ids = torch.LongTensor(batch['input_ids']).to(device)
            decoder_input_ids = torch.LongTensor([[rm.pad_id, ] + text_ids for text_ids in batch['labels']]).to(device)

            loss = loss_fn(rm(input_ids, decoder_input_ids)[1])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_sum += loss.item()

            data_iter.set_postfix(loss=loss.item())

            del batch
            torch.cuda.empty_cache()
        print(f"Reward Model Loss is {loss_sum / len(dataloader)}")
        self.loss_record.update(loss_sum / len(dataloader))
        self.rm.load_state_dict(rm.state_dict())
        self.rm.save()
        del rm
        return self.rm
