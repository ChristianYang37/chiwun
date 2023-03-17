import torch
from torch.utils.data import DataLoader, Dataset
import sacrebleu
import pandas as pd
from config import train_data_path, src, tgt
from tqdm import tqdm
from .utils import save_texts


class TranslationDataset(Dataset):
    def __init__(self):
        data = pd.read_csv(train_data_path)['translation']
        src_texts, tgt_texts = [], []
        for translation in data:
            if type(translation) == str:
                translation = eval(translation)
            src_texts.append(translation[src])
            tgt_texts.append(translation[tgt])
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, item):
        return self.src_texts[item], self.tgt_texts[item]


class RewardDataset:
    def __init__(self, sft, num_sample, batch_size=16):
        print("Sampling Model Output On Train Dataset")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = sft.to(device)
        model.eval()
        tokenizer = sft.tokenizer

        data = pd.read_csv(train_data_path)['translation']
        src_texts, tgt_texts = [], []
        for translation in data:
            if type(translation) == str:
                translation = eval(translation)
            src_texts.append(translation[src])
            tgt_texts.append(translation[tgt])

        generated_texts = []
        for i in tqdm(range(len(src_texts) // batch_size + 1)):
            queries = src_texts[i * batch_size:i * batch_size + batch_size]
            targets = tgt_texts[i * batch_size:i * batch_size + batch_size]
            batch = tokenizer.prepare_seq2seq_batch(src_texts=queries)

            batch["input_ids"] = torch.LongTensor(batch["input_ids"]).to(device)
            batch["attention_mask"] = torch.LongTensor(batch["attention_mask"]).to(device)

            prediction = model.transformer.generate(**batch, num_beams=num_sample, do_sample=True,
                                                    num_return_sequences=num_sample)
            prediction = tokenizer.batch_decode(prediction, skip_special_tokens=True)

            for j in range(batch_size):
                if j >= len(targets):
                    break
                generated_texts.append(prediction[j * num_sample:j * num_sample + num_sample] + [targets[j], ])
        del model

        bleu = sacrebleu.BLEU()
        bleu.trg_lang = 'zh'

        input_texts = []
        print("Ranking Data by BLEU")
        for i, texts in enumerate(tqdm(generated_texts)):
            rank = [[text, bleu.sentence_score(text, tgt_texts).score] for text in texts]
            rank.sort(key=lambda x: x[1], reverse=True)
            texts = [rank[i][0] for i in range(len(rank))]
            input_texts.append([src_texts[i], ] + texts)
        self.input_texts = input_texts
        save_texts(input_texts)
