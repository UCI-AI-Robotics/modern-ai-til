# Training without HuggingFace Traniner API
from datasets import load_dataset, Dataset

# dataset download
klue_ynat_train = load_dataset('klue', 'ynat', split="train")
klue_ynat_eval = load_dataset('klue', 'ynat', split="validation")

# remove useless columns
klue_ynat_train = klue_ynat_train.remove_columns(['guid', 'url', 'date'])
klue_ynat_eval = klue_ynat_eval.remove_columns(['guid', 'url', 'date'])

klue_ynat_label = klue_ynat_train.features['label']

# new column - int2str category
def make_str_label(batch):
    batch['label_str'] = klue_ynat_label.int2str(batch['label'])
    return batch

klue_ynat_train = klue_ynat_train.map(make_str_label, batched=True, batch_size=1000)
print(f"klue_ynat_train[0]: {klue_ynat_train[0]}")

# split datasets into train/test/validate
train_dataset = klue_ynat_train.train_test_split(
    test_size=10000, shuffle=True, seed=42
)['test']
# TODO: check original code and compare
test_dataset = klue_ynat_eval.train_test_split(
    test_size=1000, shuffle=True, seed=42
)['test']
valid_dataset = klue_ynat_eval.train_test_split(
    test_size=1000, shuffle=True, seed=42
)['test']

# Traninig with Trainer API
import torch
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

def tokenize_helper(ex):
    return tokenizer(ex['title'], padding='max_length',truncation=True)

model_id = "klue/roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=len(train_dataset.features['label'].names)
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

train_dataset = train_dataset.map(tokenize_helper, batched=True)
valid_dataset = valid_dataset.map(tokenize_helper, batched=True)
test_dataset = test_dataset.map(tokenize_helper, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    push_to_hub=False
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

performance = trainer.evaluate(test_dataset) # 정확도 0.84
print(f"Final Performance: {performance}")