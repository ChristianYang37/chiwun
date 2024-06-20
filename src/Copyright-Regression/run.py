import os
import torch
import argparse
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import GPT2Model, GPT2TokenizerFast, GPT2Config
from datasets import load_dataset
import math
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

random_seed_default = [2024, 3409, 121564613, 325123515, 78976857, 6543990,
                       45698975, 873886509, 12343234567898, 7101928384, 65748392190]
gamma_default = [0.1, 0.2, 0.3, 0.4, 0.5]
proportion_default = [0.1, 0.2, 0.4, 0.6, 0.8]
hf_name_or_path = "gpt2"


def get_hyper_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", default="gamma", help="which hyper-parameter to be compared")
    parser.add_argument("--lr", default=1e-4)
    parser.add_argument("--batch_size", "-bs", default=32)
    parser.add_argument("--num_epochs", "-ne", default=20)
    parser.add_argument("--output", "-o", default="results.txt")

    args = parser.parse_args()
    task = args.task
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    path = args.output
    eval_only = False

    if task == 'gamma':
        gammas = gamma_default
        proportions = [0.1]
    elif task == 'proportion':
        gammas = [0.2]
        proportions = proportion_default
    elif task == 'gpt2':
        gammas = [0.2]
        proportions = proportion_default
        num_epochs = 1
        eval_only = True
    else:
        raise ValueError(f"{task}")

    return gammas, proportions, lr, batch_size, num_epochs, path, eval_only


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    random.seed(seed)


def wikitext_ds(split: str = 'train'):
    ds = load_dataset(
        path='wikitext',
        name='wikitext-2-raw-v1',
        split=split,
        cache_dir='data',
        data_dir='data'
    )
    return ds


def get_clean_text(text, tokenizer):
    # text = train['text'] + val['text'] + test['text']
    while '' in text:
        text.remove('')
    ret = []
    print("processing data")
    for stc in tqdm(text):
        tokenized_ids = tokenizer(stc)["input_ids"]
        while len(tokenized_ids) > 0:
            ret.append(tokenizer.decode(tokenized_ids[:256]))
            tokenized_ids = tokenized_ids[256:]
    return ret


class MyDataset(Dataset):
    def __init__(self, text, cp_text, tokenizer):

        self.input_ids, self.attention_mask, self.is_copyrighted = [], [], []
        print("loading data")
        for stc in tqdm(text):
            return_dict = tokenizer(stc, padding='max_length', return_tensors='pt', truncation=True)

            self.input_ids.append(return_dict['input_ids'])
            self.attention_mask.append(return_dict['attention_mask'])
            self.is_copyrighted.append(0)

        for stc in tqdm(cp_text):
            return_dict = tokenizer(stc, padding='max_length', return_tensors='pt', truncation=True)

            self.input_ids.append(return_dict['input_ids'])
            self.attention_mask.append(return_dict['attention_mask'])
            self.is_copyrighted.append(1)

        self.input_ids = torch.cat(self.input_ids)
        self.attention_mask = torch.cat(self.attention_mask)
        self.is_copyrighted = torch.LongTensor(self.is_copyrighted)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return {
            "input_ids": self.input_ids[item],
            "attention_mask": self.attention_mask[item],
            "is_copyrighted": self.is_copyrighted[item]
        }


def my_dataloader(batch_size, cp_proportion, seed):
    tokenizer = GPT2TokenizerFast.from_pretrained(hf_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 256

    train, val, test = wikitext_ds('train'), wikitext_ds('validation'), wikitext_ds('test')
    text = get_clean_text(train['text'] + val['text'], tokenizer)
    test_text = get_clean_text(test['text'], tokenizer)

    text, cp_text = train_test_split(text, test_size=cp_proportion, random_state=seed)

    train_iter = DataLoader(MyDataset(text, cp_text, tokenizer), batch_size=batch_size, shuffle=True)

    test_iter = DataLoader(MyDataset(test_text, [], tokenizer), batch_size=batch_size, shuffle=True)
    non_cp_test_iter = DataLoader(MyDataset(text, [], tokenizer), batch_size=batch_size, shuffle=True)
    cp_test_iter = DataLoader(MyDataset([], cp_text, tokenizer), batch_size=batch_size, shuffle=True)

    return train_iter, test_iter, non_cp_test_iter, cp_test_iter


class CopyrightRegressionGPT2(nn.Module):
    def __init__(self, gamma=0.5):
        super(CopyrightRegressionGPT2, self).__init__()
        self.config = GPT2Config.from_pretrained(hf_name_or_path, cache_dir='model_weight')
        self.transformer = GPT2Model.from_pretrained(hf_name_or_path)
        self.lm_head = nn.Linear(768, self.config.vocab_size, bias=False)

        self.gamma = gamma

    def forward(self, input_ids, attention_mask, is_copyrighted):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        labels = input_ids.clone()
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = math.exp(loss.item())

        if is_copyrighted.sum() > 0:
            cp_logits = self.lm_head(hidden_states[is_copyrighted].detach())
            shift_cp_logits = cp_logits[..., :-1, :].contiguous()
            shift_cp_labels = input_ids[is_copyrighted][..., 1:].contiguous()
            loss += self.gamma / loss_fct(shift_cp_logits.view(-1, shift_cp_logits.size(-1)), shift_cp_labels.view(-1))

        return {
            "loss": loss,
            "perplexity": perplexity
        }


def fine_tune(model, data_iter, optim, device, scaler):
    model.train()

    data_iter = tqdm(data_iter)

    step_loss_list = []
    for X in data_iter:
        X["input_ids"] = X["input_ids"].to(device)
        X["attention_mask"] = X["attention_mask"].to(device)
        X["is_copyrighted"] = X["is_copyrighted"].to(device)

        optim.zero_grad()
        with autocast():
            return_dict = model(**X)

        loss = scaler.scale(return_dict["loss"]).backward()

        scaler.step(optim)
        scaler.update()

        loss = loss.item()

        data_iter.set_postfix(loss=loss)
        step_loss_list.append(loss)
    return step_loss_list


@torch.no_grad()
def evaluate(model, data_iter, device, metric='loss'):
    model.eval()

    data_iter = tqdm(data_iter)

    loss_sum = 0
    for X in data_iter:
        X["input_ids"] = X["input_ids"].to(device)
        X["attention_mask"] = X["attention_mask"].to(device)
        X["is_copyrighted"] = X["is_copyrighted"].to(device)

        return_dict = model(**X)

        loss_sum += return_dict["loss"].item()
    avg_loss = loss_sum / len(data_iter)
    if metric == 'loss':
        return avg_loss
    else:
        # metric == "perplexity"
        return math.exp(avg_loss)


def run(seed, gamma, proportion, lr, batch_size, num_epochs, eval_only):
    device = "cuda"

    train_iter, test_iter, non_cp_test_iter, cp_test_iter = my_dataloader(batch_size, proportion, seed)
    model = CopyrightRegressionGPT2(gamma=gamma).cuda()
    optim = AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()

    min_loss = 1e8
    ret_tau = 0
    test_ppl = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        if not eval_only:
            step_loss_list = fine_tune(model, train_iter, optim, device, scaler)
            avg_loss = sum(step_loss_list) / len(step_loss_list)
            print(f"loss={avg_loss}")
        else:
            avg_loss = 0

        test_perplexity = evaluate(model, test_iter, device, 'ppl')
        print(f"test_perplexity={test_perplexity}")

        non_cp_loss = evaluate(model, non_cp_test_iter, device, 'loss')
        cp_loss = evaluate(model, cp_test_iter, device, 'loss')
        print(f"tau_CE={cp_loss - non_cp_loss}")

        if avg_loss < min_loss:
            min_loss = avg_loss
            ret_tau = cp_loss - non_cp_loss
            test_ppl = test_perplexity
    return ret_tau, test_ppl


def main():
    result = dict()
    gammas, proportions, lr, batch_size, num_epochs, path, eval_only = get_hyper_params()

    for gamma in gammas:
        for proportion in proportions:
            taus, ppls = [], []
            for seed in random_seed_default:
                set_seed(seed)
                tau, ppl = run(seed, gamma, proportion, lr, batch_size, num_epochs, eval_only)
                taus.append(tau)
                ppls.append(ppl)
            result[(gamma, proportion)] = [(
                min(taus),
                sum(taus) / len(taus),
                max(taus)
            ), (
                min(ppls),
                sum(ppls) / len(ppls),
                max(ppls)
            )]
    with open(path, encoding='utf-8', mode='w') as file:
        file.write(str(result))


if __name__ == '__main__':
    main()