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
print(f"[Dataset Sample] klue_ynat_train[0]: {klue_ynat_train[0]}")

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

# tokenizing helper - tokenizing to 'title' column
def tokenize_helper(ex):
    return tokenizer(ex['title'], padding='max_length',truncation=True)

# Load model and tokenizer
model_id = "klue/roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=len(train_dataset.features['label'].names)
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# prepare dataset
train_dataset = train_dataset.map(tokenize_helper, batched=True)
valid_dataset = valid_dataset.map(tokenize_helper, batched=True)
test_dataset = test_dataset.map(tokenize_helper, batched=True)

# define Arguments for Trainer
training_args = TrainingArguments(
    output_dir="./results_w_trainer",
    num_train_epochs=2,
    # per_device_train_batch_size=4, # For RTX3070
    # per_device_eval_batch_size=4,  # For RTX3070
    per_device_train_batch_size=8, # For RTX4090
    per_device_eval_batch_size=8,  # For RTX4090
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    push_to_hub=False
)

# Compute performance - right value percentage
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

# Define Trainer with TrainingArguments
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# train model
trainer.train()

# evaluate model
performance = trainer.evaluate(test_dataset)
print(f"Final Performance: {performance}")
# Final Performance: {
#     'eval_loss': 0.647104799747467, 
#     'eval_accuracy': 0.843, 
#     'eval_runtime': 4.197, 
#     'eval_samples_per_second': 238.266, 
#     'eval_steps_per_second': 59.567, 
#     'epoch': 1.0
# }

# Specify customized model configs
id2label = {i: label for i, label in enumerate(train_dataset.features['label'].names)}
label2id = {label: i for i, label in id2label.items()}
model.config.id2label = id2label
model.config.label2id = label2id

# Push trained model onto HG hub 
from huggingface_hub import login
from hg_token import HG_TOKEN

login(token=HG_TOKEN)
repo_id = "kimsooyoung/roberta-base-klue-ynat-classification-w-trainer"

trainer.push_to_hub(repo_id) 