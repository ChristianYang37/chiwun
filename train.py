import warnings
import argparse
from models import RL, SFT, RM
from sampler import ppo_sampler, rm_sampler
from rl_bleu import rm_trainer
from ppo import ppo_trainer
from evaluate import evaluate
from config import *

warnings.filterwarnings('ignore')

paser = argparse.ArgumentParser()
paser.add_argument("--train_rm_only", action="store_true", default=False)
paser.add_argument("--train_ppo_only", action="store_true", default=False)

args = paser.parse_args()

train_rm = args.train_rm_only
train_ppo = args.train_ppo_only
if not train_rm and not train_ppo:
    train_rm, train_ppo = True, True


def train():
    rl = RL(rl_model_path)
    sft = SFT()
    rm = RM(rm_model_path)

    sft_score = evaluate(sft, sft.tokenizer)
    print(f"Before Training, BLEU of sft is {sft_score}\n")

    ppo_data_iter = ppo_sampler(batch_size)
    rm_data_iter = rm_sampler(sft, num_sample)

    rm_t = rm_trainer(rm, sft)
    ppo_t = ppo_trainer(rl, sft, sft_score, init_beta=init_beta, init_gamma=init_gamma, kl_target=kl_target,
                        k_beta=k_beta)

    if train_rm:
        print("Train RM Model")
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}")
            rm_dataloader = rm_data_iter.iterator()
            rm = rm_t.train(rm_dataloader, train_rm_lr)
    if train_ppo:
        print("Train PPO Model")
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}")
            ppo_t.train(ppo_data_iter.iterator(), rm, train_ppo_lr)


if __name__ == '__main__':
    train()
