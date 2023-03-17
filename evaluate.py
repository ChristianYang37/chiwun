import torch
import sacrebleu
import pandas as pd
from tqdm import tqdm
from config import val_data_path, src, tgt


def evaluate(model, tokenizer, batch_size=16, use_cuda=True):
    print("Calculating Model BLEU on Validation Dataset")
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    model = model.to(device)
    model.eval()

    data = pd.read_csv(val_data_path)['translation']
    src_texts, tgt_texts = [], []
    for translation in data:
        if type(translation) == str:
            translation = eval(translation)
        src_texts.append(translation[src])
        tgt_texts.append(translation[tgt])

    sys = []
    for i in tqdm(range(len(src_texts) // batch_size + 1)):
        queries = src_texts[i * batch_size:i * batch_size + batch_size]
        batch = tokenizer.prepare_seq2seq_batch(src_texts=queries)

        batch["input_ids"] = torch.LongTensor(batch["input_ids"]).to(device)
        batch["attention_mask"] = torch.LongTensor(batch["attention_mask"]).to(device)

        prediction = model.transformer.generate(**batch, num_beams=3)
        prediction = tokenizer.batch_decode(prediction, skip_special_tokens=True)

        sys.extend(prediction)

    return sacrebleu.corpus_bleu(sys, [tgt_texts], tokenize=sacrebleu.metrics.bleu.BLEU._TOKENIZER_MAP[tgt]).score

