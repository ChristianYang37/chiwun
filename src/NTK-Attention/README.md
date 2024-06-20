# Code for *Toward Infinite-Long Prefix in Transformer*

## Installation

```commandline
pip install -r requirements.txt
```

## Quick Start

We provide a clean version of our NTK-Attention in `ntk_attn_demo.py`, we give a simple example of how to use it

```python
import torch
from ntk_attn_demo import ntk_attn

bs = 8  # batch size
num_heads = 32  # number of heads in attention
seq_len = 1024  # input length
dim = 128  # model dimension
q, k, v = torch.randn(bs, num_heads, seq_len, dim), torch.randn(bs, num_heads, seq_len, dim), torch.randn(bs, num_heads, seq_len, dim)

# Trainable parameters
phi_k = torch.randn(1, num_heads, dim, 1)  # k \in \R^{d} in NTK-Attention
phi_kv = torch.randn(1, num_heads, dim, dim)  # Z \in \R^{d \times d} in NTK-Attention

# Compute the output
out = ntk_attn(q, k, v, phi_k, phi_kv, casual_mask=True)

# parameter info of ntk_attn
# def ntk_attn(query_layer, key_layer, value_layer, sum_phi_k, sum_phi_kv, causal_mask=True):
#     """
# 
#     :param query_layer: shape[bs, h, nq, d]
#     :param key_layer: shape[bs, h, nk, d]
#     :param value_layer: shape[bs, h, nk, d]
#     :param sum_phi_k: shape[1, h, d, 1]
#     :param sum_phi_kv: shape[1, h, d, d]
#     :param causal_mask: bool, default to True
#     :return: context_layer
#     """
```

## Experiment Scripts

### Experiment in Section 4.4

```commandline
python ntk-attn-error/ntk_attn_error.py
```

### Experiment to evaluate on ViT

#### FFT-ViT

````commandline
python ntk-attn-vit/fft_vit_cifar.py
````

```commandline
python ntk-attn-vit/fft_vit_food.py
```

#### NTK-ViT

```commandline
python ntk-attn-vit/ntk_vit_cifar.py
```

```commandline
python ntk-attn-vit/ntk_vit_food.py
```

### Experiment to evaluate on ChatGLM3-6B

#### P-Tuning V2

```commandline
python ntk-attn-chatglm/p_tuning_glm.py
```

Below are the hyper-parameters you should modify

```python
model_checkpoint = "THUDM/chatglm3-6b"
dataset_name = "boolq"
m = 10  # prefix length

dataset = load_dataset("super-glue", dataset_name)

dataset_features = {
    "boolq": ["question", "passage"]
}
```

#### LoRA

```commandline
python ntk-attn-chatglm/lora_glm.py
```

Below are the hyper-parameters you should modify

```python
model_checkpoint = "THUDM/chatglm3-6b"
dataset_name = "boolq"
r = 8
lora_alpha = 16
lora_dropout = 0

dataset = load_dataset("super-glue", dataset_name)

dataset_features = {
    "boolq": ["question", "passage"]
}
```

#### NTK-Attention

```commandline
python ntk-attn-chatglm/ntk_glm.py
```

Below are the hyper-parameters you should modify

```python
model_checkpoint = "THUDM/chatglm3-6b"
dataset_name = "boolq"

dataset = load_dataset("super-glue", dataset_name)

dataset_features = {
    "boolq": ["question", "passage"]
}
```
