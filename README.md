# Babel: Training Machine Translation Model with Reinforcement Learning from BLEU Score

## Introduction
大翻译模型存在着一定的过拟合问题，当使用翻译模型在特定领域（如医学、法律）小数据上微调时，极易发生过拟合问题，
这将会危害翻译模型原有的翻译能力。 鉴于RLHF（Reinforcement Learning from Human Feedback）在优化GPT上令人惊讶的表现，
我们依此提出了RLBLEU（Reinforcement Learning from BLEU Score），根据翻译的流畅程度来微调翻译模型，
在kl散度的限制下通过强化学习提高其在小数据的翻译表现而不损失其优秀的泛化性。

我们使用['Helsinki-NLP/opus-mt-en-zh'](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)在wmt19的英汉数据集上微调，模型在没有见过任何真实翻译数据的情况下，验证集的BLEU从28.15提升到了31.11。

本库代码基于pytorch和Transformers框架构建，现只兼容Transformers.MarianMTModel
## Installation
```commandline
git clone https://github.com/ChristianYang37/RL-BLEU.git
```
### Requirements
```commandline
pip install -r ./requirements.txt
``` 
## QuickStart
### Hyper-parameter
在开始训练之前，你需要在`config.py`查看各种超参数
```python
SFT_huggingface_card = 'Helsinki-NLP/opus-mt-en-zh'
task = 'wmt19'
src_tgt = 'zh-en'
train_data_path = './data/train.csv'
val_data_path = './data/val.csv'
num_data = 300000
src, tgt = 'en', 'zh'
rm_model_path = './rm_model'
rl_model_path = './rl_model'
num_sample = 8
batch_size = 4
init_beta = 0.2
init_gamma = 0.0
kl_target = 1
k_beta = 0.1
epochs = 20
train_rm_lr = 1e-5
train_ppo_lr = 1e-6
```
### Prepare Data
```commandline
python prepare_data.py
```
### Train Reward Model
我们通过BLEU指标来评判翻译程度的好坏，训练一个Reward Model来代替BLEU打分，我们将奖励Rollout展开到每个翻译token的奖励。

我们引用了InstructGPT的PairWise/RankWise训练方法，先使用SFT（Supervised fine-tuning）模型在训练数据上生成num_sample个样本，
为翻译样本排列打分后，最大化每两个样本评价的reward差值（详见`docs`）。
```commandline
python train.py --train_rm_only
```
### Train MT Model
我们使用OpenAI的PPO KL-Penalty算法来通过奖励对原有的MT模型进行强化学习（详见`docs`）。
```commandline
python train.py --train_ppo_only
```
## Reference