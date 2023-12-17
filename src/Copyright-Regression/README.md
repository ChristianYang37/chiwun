## Installation

```commandline
pip install -r requirements.txt
```

## Quickstart

Compare difference $\gamma_c$ with fixed proportion:

```commandline
python run.py --task gamma --lr 1e-4 --batch_size 32 --num_epochs 20 --output gamma_results.txt
```

Compare difference proportion with fixed $\gamma_c$:

```commandline
python run.py --task proportion --lr 1e-4 --batch_size 32 --num_epochs 20 --output proportion_results.txt
```

Evaluate the copyright protection performance of GPT-2:

```commandline
python run.py --task gpt2 --lr 1e-4 --batch_size 32 --num_epochs 20 --output eval_gpt2_results.txt
```

## Our paper

How to protect copyright data in optimization of large language models? (AAAI-2024) [[arxiv]](https://arxiv.org/abs/2308.12247)
