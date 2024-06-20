from datasets import load_dataset
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import TrainingArguments, Trainer

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import numpy as np
import evaluate
import os


def set_seed(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


set_seed()

model_checkpoint = "google/vit-base-patch16-224-in21k"

dataset = load_dataset("food101")

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label


image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


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


model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)


def phi(x):
    return nn.ELU()(x / (x.size(-1) ** 0.25)) + 1


def dropout(x):
    return nn.Dropout(0.2)(x)


class NTKViTSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.freeze()

        self.phi_k = nn.Parameter(
            torch.zeros(size=(1, self.num_attention_heads, self.attention_head_size, 1)))
        self.phi_kv = nn.Parameter(torch.zeros(size=(1, self.num_attention_heads,
                                                     self.attention_head_size,
                                                     self.attention_head_size)))

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states)).float()
        value_layer = self.transpose_for_scores(self.value(hidden_states)).float()
        query_layer = self.transpose_for_scores(mixed_query_layer).float()

        # self.freeze()

        d = query_layer.size(-1)
        # mask = torch.tril(torch.ones(bs, h, nq, nk), diagonal=nk - nq).to(query_layer.device)
        phi_q = phi(query_layer)
        phi_q = dropout(phi_q)

        A = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / (d ** 0.5)
        max_A = A.max(-1).values.unsqueeze(-1)
        exp_max_A = torch.exp(max_A)
        # exp_max_A[exp_max_A > 1e4] = 1e4
        A = torch.exp(A - max_A)

        D = A.sum(-1).unsqueeze(-1) + torch.matmul(phi_q, self.phi_k.abs().float()) / exp_max_A

        if head_mask is not None:
            A = A * head_mask.float()

        context_layer = (torch.matmul(A, value_layer) + torch.matmul(phi_q, self.phi_kv.float()) / exp_max_A) / D

        # print(query_layer.mean(), key_layer.mean(), value_layer.mean())
        # print(A.mean())
        # print((torch.matmul(phi_q, self.phi_kv.float()) / exp_max_A).mean())

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous().half()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, (A / D).half()) if output_attentions else (context_layer,)

        return outputs

    def freeze(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad = False


def freeze(model):
    for name, parameter in model.named_parameters():
        if "classifier" in name:
            continue
        parameter.requires_grad = False


def make_model(model):
    config = model.vit.config
    freeze(model)
    for i in range(config.num_hidden_layers):
        original_attn = model.vit.encoder.layer[i].attention.attention
        new_attn = NTKViTSelfAttention(config)
        new_attn.query.load_state_dict(original_attn.query.state_dict())
        new_attn.key.load_state_dict(original_attn.key.state_dict())
        new_attn.value.load_state_dict(original_attn.value.state_dict())

        # new_attn.query.requires_grad = False
        # new_attn.key.requires_grad = False
        # new_attn.value.requires_grad = False

        model.vit.encoder.layer[i].attention.attention = new_attn
    return model


model = make_model(model)
print_trainable_parameters(model)


model_name = model_checkpoint.split("/")[-1]
batch_size = 256

args = TrainingArguments(
    f"{model_name}-ntk-food101",
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
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
train_results = trainer.train()
