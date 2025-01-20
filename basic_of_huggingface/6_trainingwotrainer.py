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

# select computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_helper(ex):
    return tokenizer(ex['title'], padding='max_length',truncation=True)

# Load model and tokenizer
model_id = "klue/roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=len(train_dataset.features['label'].names)
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# convey model to device
model.to(device)

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW

# Create dataset loader wtih DataLoader from torch
def make_dataloader(dataset, batch_size, shuffle=True):
    dataset = dataset.map(tokenize_helper, batched=True).with_format("torch")
    dataset = dataset.rename_column("label", "labels") # column rename
    dataset = dataset.remove_columns(column_names=['title']) # remove useless cols
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_dataloader = make_dataloader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = make_dataloader(valid_dataset, batch_size=8, shuffle=False)
test_dataloader = make_dataloader(test_dataset, batch_size=8, shuffle=False)

# def train_epoch(model, data_loader, optimizer):
#     model.train()
#     total_loss = 0
#     for batch in tqdm(data_loader):
#         optimizer.zero_grad()
#         input_ids = batch['input_ids'].to(device) # 모델에 입력할 토큰 아이디
#         attention_mask = batch['attention_mask'].to(device) # 모델에 입력할 어텐션 마스크
#         labels = batch['labels'].to(device) # 모델에 입력할 레이블
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels) # 모델 계산
#         loss = outputs.loss # 손실
#         loss.backward() # 역전파
#         optimizer.step() # 모델 업데이트
#         total_loss += loss.item()
#     avg_loss = total_loss / len(data_loader)
#     return avg_loss

# define train helper function
def train_epoch(model, data_loader, optimizer):
    # model > train mode
    model.train()
    # define loss metric var
    total_loss = 0
    # for-loop with tqdm
    for batch in tqdm(data_loader):
        # optimizer zero grad
        optimizer.zero_grad()
        # prepare model inputs - input_ids/attention_mask/labels
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # get model output
        output = model(input_ids, attention_mask=attention_mask, labels=labels) # 모델 계산

        # get loss and run back propagation - loss/backward/step
        loss = output.loss
        loss.backward()
        optimizer.step()

        # get loss sum 
        total_loss += loss.item()

    # get avg loss
    avg_loss = total_loss / len(data_loader)
    return avg_loss

learning_rate = 5e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)


train_loss = train_epoch(model, train_dataloader, optimizer)