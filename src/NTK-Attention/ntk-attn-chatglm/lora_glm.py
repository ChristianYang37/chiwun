import os
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig
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
r = 8
lora_alpha = 16
lora_dropout = 0

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


peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
model = get_peft_model(model, peft_config)
print_trainable_parameters(model)


model_name = model_checkpoint.split("/")[-1]
batch_size = 256

args = TrainingArguments(
    f"{model_name}-lora",
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
