# __author__ == ChiWun Yang
# email == christiannyang37@gmail.com

import torch
from tqdm import tqdm
from .utils import logger
from .criterion import ObjectiveLoss
from transformers import AutoTokenizer
from config import SFT_huggingface_card
from evaluate import evaluate


class ppo_trainer:
    # example for kwargs below
    # kwargs = {
    #     'init_beta': 0.2,
    #     'init_gamma': 0.1,
    #     'kl_target': 6,
    #     'k_beta': 0.1
    # }

    def __init__(self, rl, sft, best_score, **kwargs):
        """
            see https://arxiv.org/pdf/1909.08593v2.pdf for more detail.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(SFT_huggingface_card)

        self.rl = rl  # Reinforcement learning model
        self.sft = sft  # Supervised fine-tuning model

        # ppo objective, see https://arxiv.org/pdf/2204.05862v1.pdf
        self.loss_fn = ObjectiveLoss(
            init_beta=kwargs['init_beta'],
            init_gamma=kwargs['init_gamma'],
            kl_target=kwargs['kl_target'],
            k_beta=kwargs['k_beta']
        )

        self.logger = logger()
        self.best_score = best_score

    def train(self, dataloader, rm, lr=5e-6):
        print("Train RL Model")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        optimizer = torch.optim.SGD(self.rl.parameters(), lr=lr)

        # load models in gpu
        sft = self.sft.to(device)
        rl = self.rl.to(device)
        rm = rm.to(device)

        rl.train()
        loss_sum, ce_loss_sum, beta_sum, reward_sum, kl_sum = 0, 0, 0, 0, 0
        data_iter = tqdm(dataloader)
        for src_texts, tgt_texts in data_iter:
            batch = self.tokenizer.prepare_seq2seq_batch(src_texts=src_texts, tgt_texts=tgt_texts)
            batch["input_ids"] = torch.LongTensor(batch["input_ids"]).to(device)
            batch["attention_mask"] = torch.LongTensor(batch["attention_mask"]).to(device)
            batch["labels"] = torch.LongTensor(batch["labels"]).to(device)

            sft_output = sft(**batch)
            rl_output = rl(**batch)

            reward = rm.scoring(batch['input_ids'], rl_output.logits, device).to(device)

            # calculate loss
            loss, kl, beta = self.loss_fn(rl_output.logits, sft_output.logits, reward, rl_output.loss)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            data_iter.set_postfix(loss=loss.item(), reward=float(reward.mean()), kl=float(kl))

            loss_sum += loss.item()
            ce_loss_sum += rl_output.loss.item()
            beta_sum += beta
            reward_sum += float(reward.mean())
            kl_sum += float(kl)

            del batch, sft_output, rl_output

        self.rl.load_state_dict(rl.state_dict())

        n = len(dataloader)

        print(f"PPO Train Loss is {loss_sum / n}")
        print(f"PPO Train Reward is {reward_sum / n}")
        self.logger.update(loss=loss_sum / n, ce_loss=ce_loss_sum / n, beta=beta_sum / n, reward=reward_sum / n,
                           kl=kl_sum / n)

        del rl, sft, rm
        torch.cuda.empty_cache()

        bleu = evaluate(self.rl, self.rl.tokenizer)
        print(f"BLEU is {bleu}")
        if bleu > self.best_score:
            self.best_score = bleu
            self.rl.save()
