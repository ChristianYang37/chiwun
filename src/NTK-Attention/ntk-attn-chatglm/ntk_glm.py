import os
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
import math
from transformers import TrainingArguments, Trainer
# from peft import get_peft_model
import numpy as np
import evaluate


def set_seed(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


set_seed()

model_checkpoint = "THUDM/chatglm3-6b"
dataset_name = "boolq"

dataset = load_dataset("super-glue", dataset_name)

dataset_features = {
    "boolq": ["question", "passage"]
}

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, padding_size="left")
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    inputs = ["" for _ in range(len(example_batch[dataset_features[dataset_name]]))]
    for feature in dataset_features[dataset_name]:
        texts = tokenizer.decode(tokenizer(example_batch[feature], max_length=128, truncation=True)["input_ids"])
        for i in range(len(inputs)):
            inputs[i] += feature + " : " + texts[i]
    inputs = tokenizer(inputs, truncation=True, padding=128 * len(dataset_features[dataset_name]), return_tensors="pt")
    inputs["labels"] = torch.LongTensor([label2id[label] for label in example_batch["label"]])
    return inputs


def preprocess_val(example_batch):
    """Apply train_transforms across a batch."""
    inputs = ["" for _ in range(len(example_batch[dataset_features[dataset_name]]))]
    for feature in dataset_features[dataset_name]:
        texts = tokenizer.decode(tokenizer(example_batch[feature], max_length=128, truncation=True)["input_ids"])
        for i in range(len(inputs)):
            inputs[i] += feature + " : " + texts[i]
    inputs = tokenizer(inputs, truncation=True, padding=128 * len(dataset_features[dataset_name]), return_tensors="pt")
    inputs["labels"] = torch.LongTensor([label2id[label] for label in example_batch["label"]])
    return inputs


train_ds = dataset["train"]
val_ds = dataset["validation"]

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            print(_)
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    trust_remote_code=True,
    empty_init=False,
    use_cache=False
)


def phi(x):
    return nn.ELU()(x / (x.size(-1) ** 0.25)) + 1


def ntk_attn(query_layer, key_layer, value_layer, sum_phi_k, sum_phi_kv, causal_mask=True):
    """

    :param query_layer: shape[bs, h, nq, d]
    :param key_layer: shape[bs, h, nk, d]
    :param value_layer: shape[bs, h, nk, d]
    :param sum_phi_k: shape[1, h, d, 1]
    :param sum_phi_kv: shape[1, h, d, d]
    :param causal_mask: bool, default to True
    :return: context_layer
    """
    dtype = query_layer.dtype

    query_layer, key_layer, value_layer = query_layer.float(), key_layer.float(), value_layer.float()
    k = sum_phi_k.abs().float()
    Z = sum_phi_kv.float()

    bs, h = query_layer.size()[:2]
    nq, nk = query_layer.size(-2), key_layer.size(-2)
    d = query_layer.size(-1)

    if causal_mask:
        mask = torch.tril(torch.ones(bs, h, nq, nk), diagonal=nk - nq).to(query_layer.device)
    else:
        mask = None

    phi_q = phi(query_layer)

    A = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / (d ** 0.5)
    max_A = A.max(-1).values.unsqueeze(-1)
    exp_max_A = torch.exp(max_A)
    A = torch.exp(A - max_A)

    if mask is not None:
        A *= mask

    D = A.sum(-1).unsqueeze(-1) + torch.matmul(phi_q, k) / exp_max_A
    context_layer = (torch.matmul(A, value_layer) + torch.matmul(phi_q, Z) / exp_max_A) / D

    return context_layer.contiguous().to(dtype)


class CoreAttention(nn.Module):
    def __init__(self, config, layer_number):
        super(CoreAttention, self).__init__()

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = nn.Dropout(config.attention_dropout)

        self.phi_k = nn.Parameter(
            torch.randn(size=(1, config.num_attention_heads, self.hidden_size_per_attention_head, 1)) / 10000)
        self.phi_kv = nn.Parameter(torch.randn(size=(1, config.num_attention_heads,
                                                     self.hidden_size_per_attention_head,
                                                     self.hidden_size_per_attention_head)) / 10000)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3).float() for k in
                                               [query_layer, key_layer, value_layer]]
        query_layer, key_layer, value_layer = query_layer.contiguous(), key_layer.contiguous(), value_layer.contiguous()
        context_layer = ntk_attn(query_layer, key_layer, value_layer, self.phi_k, self.phi_kv).permute(2, 0, 1, 3)

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)
        return context_layer


def freeze(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False


def make_model(model, ):
    config = model.config
    freeze(model)
    for i in range(28):
        attn = CoreAttention(config, i).to(model.device)
        model.transformer.encoder.layers[i].self_attention.core_attention = attn
    return model


model = make_model(model)
print_trainable_parameters(model)


model_name = model_checkpoint.split("/")[-1]
batch_size = 256

args = TrainingArguments(
    f"{model_name}-ntk",
    remove_unused_columns=False,
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=1e-3,
    # weight_decay=1e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    num_train_epochs=30,
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    label_names=["labels"],
)


metric = evaluate.load("accuracy")


# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    return examples


trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
train_results = trainer.train()
